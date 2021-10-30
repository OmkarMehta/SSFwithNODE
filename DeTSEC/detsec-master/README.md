# DETSEC

This is the repository associated to the DETSEC model (Deep Multivariate Time Series EmbeddingClustering via Attentive-Gated Autoencoder) published at PAKDD 2020 [1].



The main file: DETSEC.py takes as input 3 different parameters:
- The directory in which data are stored
- The number of attribute/dimensions on which multivariate time series are defined
- The number of clusters

Exmamples:

python DETSEC.py ECG 2 2

Here we run the DETSEC model on the ECG datasets that has multivariate time series with dimension equal to 2 and with a number of expected clusters equal to 2 too.
The directory ECG contains two files:
- data.npy
- seq_length.npy

The first file data.npy contains the multivariate time series organized taking into account the maximum time series length of the dataset
The secondo file seq_length.npy contains as many rows as the data.npy. Each row contains the length (sequence length * dimensions) corresponding to the assocaited multivariate time series with positional reference to the data.npy file

For instance, the ECG dataset has 200 multivariate time series with 2 dimensions varying from 39 to 152.
The data.npy will have a shape of (200,304) since there are 200 multivariate time series with a maximum length of 152, each timestamps with 2 dimensions.
If the seq_length.npy file contains the value 96, the corresponding time series will have 48 valid timestamps (48 * 2) with the rest of the row padded with zero values (104 * 2)

The DETSEC.py will write two files:
- detsec_features.npy: the representation learnt by the DETSEC model
- detsec\_clust\_assignment.npy: the clustering assignment obtained using the K-Means algorithm on the learnt representation

The code was tested considering:
- tensorflow-gpu==1.14.0
- scikit-learn==0.22.2.post1
- scipy==1.4.1
- numpy==1.18.4
- gast==0.2.2



[1] Dino Ienco, Roberto Interdonato: Deep Multivariate Time Series Embedding Clustering via Attentive-Gated Autoencoder. PAKDD (1) 2020: 318-329