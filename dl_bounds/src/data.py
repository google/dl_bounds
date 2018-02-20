# coding=utf-8
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset retrieval."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle
import os
from dl_bounds.src.exp_helpers import flip_labels
from dl_bounds.src.exp_helpers import get_split
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def get_mnist(data_path, val_size=10000):
  ds = tf.contrib.learn.datasets.mnist.read_data_sets(
      data_path, one_hot=False, validation_size=val_size, seed=1)

  return (ds.train.images, ds.train.labels, ds.validation.images,
          ds.validation.labels)


def get_cifar10(data_path):
  """Returns cifar10 dataset.

  Args:
     data_path: dataset location.

  Returns:
    tuple (training instances, training labels,
           testing instances, testing labels)
    Instances of dimension # of instances X dimension.
  """
  x_train = np.zeros((50000, 3072))
  y_train = np.zeros((50000,), dtype=int)

  x_val = np.zeros((10000, 3072))
  y_val = np.zeros((10000,), dtype=int)

  cur = 0
  for batch_index in range(1, 6):
    with tf.gfile.Open(
        os.path.join(data_path,
                     "cifar-10-batches-py/data_batch_%d" % batch_index),
        "rb") as fo:
      batch_data = cPickle.load(fo)
      m = batch_data["data"].shape[0]
      x_train[cur:cur + m, :] = batch_data["data"].astype(np.float32)
      y_train[cur:cur + m] = np.array(batch_data["labels"])
      cur += m

  assert cur == 50000

  with tf.gfile.Open(
      os.path.join(data_path, "cifar-10-batches-py/test_batch"), "rb") as fo:
    batch_data = cPickle.load(fo)
    x_val = batch_data["data"].astype(np.float32)
    y_val = np.array(batch_data["labels"])

  x_train /= 255.0
  x_val /= 255.0

  return (x_train, y_train, x_val, y_val)


def get_data(dataset_name, data_path, split_i, split_n, flip_label_ratio=0):
  """Returns a dataset or a given split.

  Args:
     dataset_name: possible choice: cifar10, mnist, covtype.
     data_path: dataset location.
     split_i: split index.
     split_n: number of examples per split. If -1 -- returns the whole dataset.
     flip_label_ratio: randomly flips given amount of labels in the
                       training and testing sets.

  Returns:
     tuple (training instances, training labels,
            testing instances, testing labels)
     Instances of dimension # of instances X dimension.
  """

  if dataset_name == "mnist":
    (x, y, _, _) = get_mnist(data_path)
    # Subsampling valdation set from the training set
    # (concerned that val follows a sligtly different distribution)
    x_train, x_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=1)

  elif dataset_name == "cifar10":
    (x_train, y_train, x_val, y_val) = get_cifar10(data_path)

  elif dataset_name == "covtype":
    with tf.gfile.Open(os.path.join(data_path, "covtype.mat"), "r") as fh:
      mat = loadmat(fh)
      x, y = mat["data"].T.todense(), mat["label"].squeeze()
      y -= 1
      StandardScaler(copy=False, with_mean=True, with_std=True).fit_transform(x)
      x_train, x_val, y_train, y_val = train_test_split(
          x, y, test_size=0.33, random_state=1)

  if split_n > 0:  # For negative split_n, return all the data
    x_train, y_train = get_split(x_train, y_train, split_i, split_n)

  num_classes = len(set(y_train))

  if flip_label_ratio > 0:
    tf.logging.info("Flipping %f%% of labels in the training set",
                    flip_label_ratio * 100)
    y_train = flip_labels(y_train, flip_label_ratio)
    y_val = flip_labels(y_val, flip_label_ratio)

  assert (y_train.min() == 0) and (y_val.min() == 0)

  lb = LabelBinarizer()
  y_train = lb.fit_transform(y_train)
  y_val = lb.transform(y_val)

  return (x_train, y_train, x_val, y_val, num_classes)


class LocalDatasetProvider(object):
  """Data provider for an in-memory dataset."""

  def __init__(self, x, y, limit_size=-1, shuffle_seed=1):
    self.x = x
    self.y = y
    self.index = None
    self.rand = None
    self.reset_and_reshuffle(shuffle_seed)
    self.limit_size(limit_size)

  def get_input_dim(self):
    return self.x.shape[1]

  def get_output_dim(self):
    return self.y.shape[1]

  def has_more(self):
    return self.cur < self.size

  def read_next(self, n):
    if self.cur <= self.size:
      n_read = min(self.size - self.cur, n)
      x_mb = self.x[self.index[self.cur:self.cur + n_read], :]
      y_mb = self.y[self.index[self.cur:self.cur + n_read], :]

      leave_out_indices = np.where(y_mb[:, 0] == -1)[0]
      if leave_out_indices:
        x_mb = np.delete(x_mb, leave_out_indices, axis=0)
        y_mb = np.delete(y_mb, leave_out_indices, axis=0)

      n_read = x_mb.shape[0]
      self.cur += n_read

      return x_mb, y_mb
    else:
      raise Exception("End-of-dataset.")

  def limit_size(self, new_size):
    if new_size != -1:
      self.size = new_size
    else:
      self.size = self.x.shape[0]

  def reset(self):
    self.cur = 0

  def reset_and_reshuffle(self, shuffle_seed):
    self.cur = 0
    self.index = np.arange(self.x.shape[0])
    self.rand = np.random.RandomState(shuffle_seed)
    self.rand.shuffle(self.index)
