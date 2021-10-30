import sys
import os
import numpy as np
import math
from operator import itemgetter, attrgetter, methodcaller
import tensorflow as tf
from tensorflow.contrib import rnn
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.contrib.rnn import DropoutWrapper
import time
import calendar
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from tensorflow.contrib import rnn
from scipy.spatial import distance
from operator import itemgetter
import random as rand
from sklearn.cluster import KMeans

def buildMaskBatch(batch_seql, max_size):
	mask_batch = []
	for el in batch_seql:
		mask_batch.append(  np.concatenate( (np.ones(el), np.zeros(max_size - el)) ) )
	return np.array(mask_batch)

def extractFeatures(ts_data, seq_length, mask_val):
	batchsz = 1024
	iterations = int(ts_data.shape[0] / batchsz)
	if ts_data.shape[0] % batchsz != 0:
	    iterations+=1
	features = None

	for ibatch in range(iterations):
		batch_data, batch_seqL = getBatch(ts_data, seq_length, ibatch, batchsz)
		batch_mask, _ = getBatch(mask_val, mask_val, ibatch, batchsz)
		partial_features = sess.run(embedding,feed_dict={input_t:batch_data, seqL:batch_seqL, mask: batch_mask})
		if features is None:
			features = partial_features
		else:
			features = np.vstack((features, partial_features))

		del batch_data
		del batch_mask
	return features

def gate(vec):
	mask = tf.layers.dense(vec, vec.get_shape()[1].value, activation=tf.sigmoid)
	return mask

def gating(outputs_list, mask):
	gating_results = None
	if mask is None:
		for i in range(len(outputs_list)):
			val = outputs_list[i]
			multiplication = val * gate(val)
			if gating_results is None:
				gating_results = multiplication
			else:
				gating_results = gating_results + multiplication
		return gating_results

	for i in range(len(outputs_list)):
		val = outputs_list[i]
		multiplication = val * gate(val)
		multiplication = tf.transpose( multiplication )
		multiplication = multiplication * mask[:,i]
		multiplication = tf.transpose( multiplication )
		if gating_results is None:
			gating_results = multiplication
		else:
			gating_results = gating_results +multiplication

	return gating_results


def attention(outputs_list, nunits, attention_size):
	outputs = tf.stack(outputs_list, axis=1)

	# Trainable parameters
	W_omega = tf.Variable(tf.random_normal([nunits, attention_size], stddev=0.1))
	b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
	u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

	# Applying fully connected layer with non-linear activation to each of the B*T timestamps;
	#  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
	v = tf.tanh(tf.tensordot(outputs, W_omega, axes=1) + b_omega)
	# For each of the timestamps its vector of size A from `v` is reduced with `u` vector
	vu = tf.tensordot(v, u_omega, axes=1)   # (B,T) shape
	alphas = tf.nn.softmax(vu)              # (B,T) shape also

	output = tf.reduce_sum(outputs * tf.expand_dims(alphas, -1), 1)
	output = tf.reshape(output, [-1, nunits])
	return output


def getBatch(X, Y, i, batch_size):
    start_id = i*batch_size
    end_id = min( (i+1) * batch_size, X.shape[0])
    batch_x = X[start_id:end_id]
    batch_y = Y[start_id:end_id]
    return batch_x, batch_y



def AE3(x, b_size, n_dim, seqL, mask, toReuse):
	with tf.variable_scope("ENCDEC",reuse=toReuse):
		n_splits = x.get_shape()[1].value / n_dim
		n_splits = int(n_splits)
		x_list = tf.split(x, n_splits, axis=1)
		x_list_bw = tf.stack(x_list[::-1],axis=1)
		x_list = tf.stack(x_list,axis=1)
		#FIXED TO 512 for big dataset
		#FOR SMALL DATASET WE RECOMMEND 64 OR 32
		nunits = 512
		outputsEncLFW = None
		outputsEncLBW = None

		with tf.variable_scope("encoderFWL", reuse=toReuse):
			cellEncoderFW = rnn.GRUCell(nunits)
			outputsEncLFW,_=tf.nn.dynamic_rnn(cellEncoderFW, x_list, sequence_length = seqL, dtype="float32")

		with tf.variable_scope("encoderBWL", reuse=toReuse):
			cellEncoderBW = rnn.GRUCell(nunits)
			outputsEncLBW,_=tf.nn.dynamic_rnn(cellEncoderBW, x_list_bw, sequence_length = seqL, dtype="float32")

		final_list_fw = []
		for i in range( n_splits ):
			final_list_fw.append( outputsEncLFW[:,i,:] )

		final_list_bw = []
		for i in range( n_splits ):
			final_list_bw.append( outputsEncLBW[:,i,:] )

		encoder_fw = attention(final_list_fw, nunits, nunits)
		encoder_bw = attention(final_list_bw, nunits, nunits)
		encoder = gate(encoder_fw) * encoder_fw + gate(encoder_bw) * encoder_bw

		x_list2decode = []
		x_list2decode_bw = []
		for i in range(n_splits):
			x_list2decode.append( tf.identity(encoder) )
			x_list2decode_bw.append( tf.identity(encoder) )

		x_list2decode = tf.stack(x_list2decode, axis=1)
		x_list2decode_bw = tf.stack(x_list2decode_bw, axis=1)

		with tf.variable_scope("decoderG", reuse=toReuse):
			cellDecoder = rnn.GRUCell(nunits)
			outputsDecG,_=tf.nn.dynamic_rnn(cellDecoder, x_list2decode, sequence_length = seqL, dtype="float32")

		with tf.variable_scope("decoderGFW", reuse=toReuse):
			cellDecoder = rnn.GRUCell(nunits)
			outputsDecGFW,_=tf.nn.dynamic_rnn(cellDecoder, x_list2decode_bw, sequence_length = seqL, dtype="float32")

		out_list = []
		out_list_bw = []
		for i in range(n_splits):
			temp_cell = outputsDecG[:,i,:]
			tt = tf.layers.dense(temp_cell, n_dim, activation=None)
			out_list.append( tt )

			temp_cell2 = outputsDecGFW[:,i,:]
			tt2 = tf.layers.dense(temp_cell, n_dim, activation=None)
			out_list_bw.append(tt2)

		reconstruct = tf.concat(out_list, axis=1)
		reconstruct2 = tf.concat(out_list_bw[::1], axis=1)

		return reconstruct, reconstruct2, encoder

#directory in which data file are stored
dirName = sys.argv[1]
#number of dimensions of the multivariate time series
n_dims = int( sys.argv[2] )
#Num of clusters, commonly, equals to the number of classes on which the dataset is defined on
n_clusters = int(sys.argv[3])

output_dir = dirName.split("/")[-1]
#DATA FILE with size:	(nSamples, (n_dims * max_length) )
dataFileName = dirName+"/data.npy"
#SEQUENCE LENGTH FILE with size: ( nSamples, )
#It contains the sequence length (multiplied by n_dims) for each sequence with positional reference to the data.npy file
#This means that, if a time series has 4 attributes and it has a lenght equal to 20, the corresponding values in the seq_length.npy file will be 80
seqLFileName = dirName+"/seq_length.npy"

data = np.load(dataFileName)
n_row = data.shape[0]
n_col = data.shape[1]

seqLength = np.load(seqLFileName)

orig_data = data
orig_seqLength = seqLength
n_feat = data.shape[1]

b_size = tf.placeholder(tf.float32, () , name="b_size")
dropOut = tf.placeholder(tf.float32, () , name="dropOut")
seqL = tf.placeholder(tf.float32, (None) , name="seqL")
input_t = tf.placeholder(tf.float32, (None, n_feat), name='inputs')
mask = tf.placeholder(tf.float32, (None, n_feat), name='mask')
target_t = tf.placeholder(tf.float32, (None, n_feat), name='target_t')

sess = tf.InteractiveSession()

reconstruction, reconstruction2, embedding = AE3(input_t, b_size, n_dims, seqL, mask, False)


b_centroids = tf.placeholder(tf.float32, (None, embedding.get_shape()[1].value), name='b_centroids')
loss_fw = tf.square( (target_t - reconstruction) * mask )
loss_fw = tf.reduce_sum(loss_fw, axis=1)

loss_bw = tf.square( (target_t - reconstruction2) * mask )
loss_bw = tf.reduce_sum(loss_bw, axis=1)

cost = tf.reduce_mean(loss_fw) + tf.reduce_mean(loss_bw)  #+ latent_loss
opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost)

################# CLUSTERING REFINEMENT CENTROIDS #################

loss_crc = tf.reduce_sum(tf.square( embedding - b_centroids ), axis=1)
loss_crc = tf.reduce_mean(loss_crc)

cost_crc = loss_crc + cost
opt_crc = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost_crc)

tf.global_variables_initializer().run()

batchsz = 16
hm_epochs = 300

iterations = int(data.shape[0] / batchsz)
max_length = data.shape[1]

if data.shape[0] % batchsz != 0:
    iterations+=1

best_loss = sys.float_info.max
noise_factor = 0.01

th = 50#number of epochs for the autoencoder pretraining step
new_centroids = None
kmeans_labels = None

for e in range(hm_epochs):
	start = time.time()
	lossi = 0
	data, seqLength = shuffle(data, seqLength, random_state=0)
	costT = 0
	costT2 = 0
	if e < th:
		data, seqLength = shuffle(data, seqLength, random_state=0)
	else:
		mask_val = buildMaskBatch(seqLength, max_length)
		features = extractFeatures(data, seqLength, mask_val)
		kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random.randint(1,10000000)).fit(features)
		new_centroids = kmeans.cluster_centers_
		kmeans_labels = kmeans.labels_
		embeddings_data = extractFeatures(data, seqLength, mask_val)
		data, seqLength, kmeans_labels = shuffle(data, seqLength, kmeans_labels, random_state=0)

	for ibatch in range(iterations):
		batch_data, batch_seql = getBatch(data, seqLength, ibatch, batchsz)
		mask_batch = buildMaskBatch(batch_seql, batch_data.shape[1])
		cost_L = 0

		#PRETRAINING ENCODER for 50 EPOCHS
		if e < th:
			_, cost_L= sess.run([opt, cost],feed_dict={ input_t:batch_data,
														target_t:batch_data,
														b_size: batch_data.shape[0],
														seqL: batch_seql,
														mask: mask_batch
														})
			cost_C=0
		#COMBINED TRAINING WITH ENCO/DEC + CLUSTERING REFINEMENT
		else:
			batch_km_labels, _ = getBatch(kmeans_labels, kmeans_labels, ibatch, batchsz)
			batch_centroids = []
			for el in batch_km_labels:
				batch_centroids.append( new_centroids[el]  )
			batch_centroids = np.array(batch_centroids)
			_, cost_L, cost_C = sess.run([opt_crc, cost, loss_crc], feed_dict={
													input_t:batch_data,
													target_t:batch_data,
													#centroids: centroids_val,
													b_size: int(batch_data.shape[0]),
													seqL: batch_seql,
													mask: mask_batch,
													b_centroids: batch_centroids
													})

		costT+=cost_L
		costT2+=cost_C
		del batch_data
		del batch_seql
		del mask_batch

	mask_val = buildMaskBatch(seqLength, max_length)
	embedd = extractFeatures(data, seqLength, mask_val)
	kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedd)
	print("Epoch:",e,"| COST_EMB:",costT/iterations," | COST_CRC: ", costT2/iterations)

output_dir = output_dir+"_detsec512"
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

mask_val = buildMaskBatch(orig_seqLength, max_length)
embedd = extractFeatures(orig_data, orig_seqLength, mask_val)
kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(embedd)

#SAVE THE DATA REPRESENTATION
np.save("detsec_features.npy", embedd )
#SAVE THE CLUSTERING ASSIGNMENT
np.save("detsec_clust_assignment.npy", np.array(kmeans.labels_) )
