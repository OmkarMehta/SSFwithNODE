###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Authors: Yulia Rubanova and Ricky Chen
###########################

import os
import numpy as np
import torch
from lib.utils import get_dict_template
import lib.utils as utils
from torchvision.datasets.utils import download_url

class HopperPhysics(object):

  T = 200
  print(f"We are inside HopperPhysics class")
  print(f"T is {T}")
  D = 14
  print("dim is {D}")

  n_training_samples = 10000
  print(f"n_training_samples is {n_training_samples}")

  training_file = 'training.pt'
  print(f"training_file name is {training_file}")

  def __init__(self, root, download = True, generate=False, device = torch.device("cpu")):
    print(f"root is {root}")
    self.root = root
    if download:
      print(f"Let's download")
      self._download()

    if generate:
      self._generate_dataset()

    if not self._check_exists():
      raise RuntimeError('Dataset not found.' + ' You can use download=True to download it')

    data_file = os.path.join(self.data_folder, self.training_file)
    print(f"data_file is {data_file}")
    print(f"Using torch.Tensor(torch.load(data_file)).to(device) to get self.data")
    self.data = torch.Tensor(torch.load(data_file)).to(device) # https://pytorch.org/docs/stable/generated/torch.load.html
    print(f"self.data.shape is {self.data.shape}")
    print("\n")
    self.data, self.data_min, self.data_max = utils.normalize_data(self.data)

    self.device =device

  def visualize(self, traj, plot_name = 'traj', dirname='hopper_imgs', video_name = None):
    r"""Generates images of the trajectory and stores them as <dirname>/traj<index>-<t>.jpg"""

    T, D = traj.size()

    traj = traj.cpu() * self.data_max.cpu() +  self.data_min.cpu()

    try:
      from dm_control import suite  # noqa: F401
    except ImportError as e:
      raise Exception('Deepmind Control Suite is required to visualize the dataset.') from e

    try:
      from PIL import Image  # noqa: F401
    except ImportError as e:
      raise Exception('PIL is required to visualize the dataset.') from e

    def save_image(data, filename):
      im = Image.fromarray(data)
      im.save(filename)

    os.makedirs(dirname, exist_ok=True)

    env = suite.load('hopper', 'stand')
    physics = env.physics

    for t in range(T):
      with physics.reset_context():
        physics.data.qpos[:] = traj[t, :D // 2]
        physics.data.qvel[:] = traj[t, D // 2:]
      save_image(
				physics.render(height=480, width=640, camera_id=0),
				os.path.join(dirname, plot_name + '-{:03d}.jpg'.format(t))
			)

  def _generate_dataset(self):
    if self._check_exists():
      return
    os.makedirs(self.data_folder, exist_ok=True)
    print('Generating dataset...')
    train_data = self._generate_random_trajectories(self.n_training_samples)
    torch.save(train_data, os.path.join(self.data_folder, self.training_file))

  def _download(self):
    if self._check_exists():
      return

    print("Downloading the dataset [325MB] ...")
    os.makedirs(self.data_folder, exist_ok=True)
    print(f"{self.data_folder} is generated")
    url = "http://www.cs.toronto.edu/~rtqichen/datasets/HopperPhysics/training.pt"
    print(f"Downloading data from this url: {url}")
    print("\n")
    download_url(url, self.data_folder, "training.pt", None)  # from torchvision.datasets.utils import download_url

  def _generate_random_trajectories(self, n_samples):

    try:
      from dm_control import suite  # noqa: F401
    except ImportError as e:
      raise Exception('Deepmind Control Suite is required to generate the dataset.') from e

    env = suite.load('hopper', 'stand')
    physics = env.physics

		# Store the state of the RNG to restore later.
    st0 = np.random.get_state()
    np.random.seed(123)

    data = np.zeros((n_samples, self.T, self.D))
    for i in range(n_samples):
      with physics.reset_context():
				# x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
        physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
        physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
        physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)
      for t in range(self.T):
        data[i, t, :self.D // 2] = physics.data.qpos
        data[i, t, self.D // 2:] = physics.data.qvel
        physics.step()

		# Restore RNG.
    np.random.set_state(st0)
    return data

  def _check_exists(self):
    print(f"we will check os.path.exists(os.path.join(self.data_folder, self.training_file)). If it exists, return")
    return os.path.exists(os.path.join(self.data_folder, self.training_file))

  @property
  def data_folder(self):
    return os.path.join(self.root, self.__class__.__name__)

	# def __getitem__(self, index):
	#     return self.data[index]

  def get_dataset(self):
    return self.data

  def __len__(self):
    return len(self.data)

  def size(self, ind = None):
    if ind is not None:
      return self.data.shape[ind]
    return self.data.shape
			
  def __repr__(self):
    fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
    fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
    fmt_str += '    Root Location: {}\n'.format(self.root)
    return fmt_str

