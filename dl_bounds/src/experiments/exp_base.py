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

"""Implements experimental logic."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import copy
import cPickle as pkl
import math
import os
from dl_bounds.src.data import get_data
from dl_bounds.src.exp_helpers import is_power2
from dl_bounds.src.models import ALL_MODELS
from dl_bounds.src.models import MLPCustom
import numpy as np
import tensorflow as tf


class Experiment(object):
  """Base class for all experiments, implements basic measurements."""

  def __init__(self, conf):
    self.conf = conf

    n_records = self.get_n_records()
    assert n_records > 0
    self.is_persistent_experiment = True

    # Generic parameters recorded for all types of experiments
    self.pass_numbers = np.zeros((n_records,), dtype=int)
    self.train_error = np.zeros((n_records,))
    self.train_zero_one_error = np.zeros((n_records,))
    self.train_grad_norm = np.zeros((n_records,))
    self.val_error = np.zeros((n_records,))
    self.val_zero_one_error = np.zeros((n_records,))
    self.val_grad_norm = np.zeros((n_records,))
    self.train_dataset_size = np.zeros((n_records,))

    self.soft_margin_eps_range = [0.001, 0.01, 0.1]
    self.soft_margins = np.zeros((n_records, len(self.soft_margin_eps_range)))

    self.weight_variance = np.zeros((n_records,))
    self.weight_entropy = np.zeros((n_records,))

    self.init_weights = None

  def get_n_records(self):
    if self.conf.log2_snapshots:
      if not is_power2(self.conf.passes):
        raise ValueError("Number of passes must be power of 2.")

      return int(math.log(self.conf.passes, 2)) + 2
    else:
      return int(self.conf.passes / self.conf.snapshot_every_n_passes) + 1

  def measure_on_train_val(self, train_dataset, val_dataset,
                           pass_index, pass_number, model):
    """Performs and memorizes generic measurements on the dataset.

    Args:
      train_dataset: Data provider for the training set.
      val_dataset: Data provider for the validations set.
      pass_index: Index of epoch.
      pass_number: Number of epoch (can differ from index,
                 e.g. pass_number is 2**pass_index).
      model: Model to evaluate on the provided data.
    """
    self.pass_numbers[pass_index] = pass_number
    self.val_error[pass_index] = model.mean_loss_on_dataset(val_dataset)
    self.val_zero_one_error[pass_index] = 1.0 - model.accuracy_on_dataset(
        val_dataset)
    self.val_grad_norm[pass_index] = np.linalg.norm(
        model.compute_gradient_on_dataset(val_dataset))

    self.train_error[pass_index] = model.mean_loss_on_dataset(train_dataset)
    self.train_zero_one_error[pass_index] = (
        1.0 - model.accuracy_on_dataset(train_dataset))
    self.train_grad_norm[pass_index] = np.linalg.norm(
        model.compute_gradient_on_dataset(train_dataset))

    self.train_dataset_size[pass_index] = train_dataset.size

    if self.conf.pflug_diagnostic:
      self.pflug_diagnostic_track = model.pflug_diagnostic_track
      self.pflug_cos_diagnostic_track = model.pflug_cos_diagnostic_track

    self.soft_margins[pass_index, :] = model.eps_margin(
        train_dataset, self.soft_margin_eps_range)

    self.weight_variance[pass_index] = model.weight_variance()
    self.weight_entropy[pass_index] = model.weight_entropy()

  def report_train_val(self, pass_index):
    """Reports various generic measurements on training and validation sets.

    Args:
      pass_index: Index of epoch.
    """
    tf.logging.info("Passes: %s", ", ".join(
        [str(x) for x in self.pass_numbers[:pass_index + 1]]
    ))
    tf.logging.info("Train error: %s", ", ".join(
        ["%.2f" % x for x in self.train_error[:pass_index + 1]]
    ))
    tf.logging.info("Train 0/1 error: %s", ", ".join(
        ["%.2f" % x for x in self.train_zero_one_error[:pass_index + 1]]
    ))
    tf.logging.info("||E^train nabla f(w_T)||: %s", ", ".join(
        ["%.2f" % x for x in self.train_grad_norm[:pass_index + 1]]
    ))

    tf.logging.info("Val error: %s", ", ".join(
        ["%.2f" % x for x in self.val_error[:pass_index + 1]]
    ))
    tf.logging.info("Val 0/1 error: %s", ", ".join(
	       ["%.2f" % x for x in self.val_zero_one_error[:pass_index + 1]]
    ))
    tf.logging.info("||E^val nabla f(w_T)||: %s", ", ".join(
        ["%.2f" % x for x in self.val_grad_norm[:pass_index + 1]]
    ))

    tf.logging.info("Weight var.: %s", ", ".join(
        ["%.2f" % x for x in self.weight_variance[:pass_index + 1]]
    ))
    tf.logging.info("Weight entropy: %s", ", ".join(
        ["%.2f" % x for x in self.weight_entropy[:pass_index + 1]]
    ))

  def get_train_val_measurements(self):
    """Returns a dict with basic measurements registered so far."""
    measurements = dict(
        pass_numbers=self.pass_numbers,
        val_error=self.val_error,
        val_zero_one_error=self.val_zero_one_error,
        val_grad_norm=self.val_grad_norm,
        train_error=self.train_error,
        train_zero_one_error=self.train_zero_one_error,
        train_grad_norm=self.train_grad_norm,
        train_dataset_size=self.train_dataset_size,
        soft_margin_eps_range=self.soft_margin_eps_range,
        soft_margins=self.soft_margins,
        weight_variance=self.weight_variance,
        weight_entropy=self.weight_entropy)

    if self.conf.pflug_diagnostic:
      measurements["pflug_diagnostic"] = np.array(self.pflug_diagnostic_track)
      measurements["pflug_cos_diagnostic"] = np.array(
          self.pflug_cos_diagnostic_track)

    return measurements

  def create_model(self,
                   input_size=None,
                   output_size=None,
                   dataset_size=-1):
    """Creates model and initializes optimizer.

    Args:
      input_size: Input dimension.
      output_size: Output dimension.
      dataset_size: Expected size of the training set (if exp decay is used).

    Returns:
      initialized model (as defined in models.py)
    """
    tf.logging.info("Creating network \"%s\" with seed %d...",
                    self.conf.network, self.conf.network_seed)

    if (self.conf.network.startswith("mlp_")
        and (self.conf.dataset, self.conf.network) not in ALL_MODELS):
      hidden_layer_sizes = map(int, self.conf.network[4:].split("-"))
      model = MLPCustom(input_size, output_size, hidden_layer_sizes)
    else:
      model = ALL_MODELS[(self.conf.dataset, self.conf.network)]()

    if self.conf.exp_decay:
      if self.conf.split_n == -1:
        pass_length = self.conf.batch_size
      else:
        pass_length = self.conf.split_n

      global_step = tf.contrib.framework.get_or_create_global_step()

      num_batches_per_epoch = dataset_size / float(pass_length)
      decay_steps = int(num_batches_per_epoch * self.conf.num_passes_per_decay)
      learning_rate = tf.train.exponential_decay(
          self.conf.learning_rate,
          global_step,
          decay_steps,
          self.conf.learning_rate_decay_factor,
          staircase=True)

      tf.logging.info("""
Using exponential learning rate decay schedule:
initial learning rate = %s
number of passes per decay = %s
learning rate decay factor = %s
      """, self.conf.learning_rate, self.conf.num_passes_per_decay,
                      self.conf.learning_rate_decay_factor)

    else:
      learning_rate = self.conf.learning_rate

    model.initialize(
        init_stddev=self.conf.init_stddev,
        learning_rate=learning_rate,
        seed=self.conf.network_seed,
        optimizer=self.conf.optimizer,
        pflug_diagnostic=self.conf.pflug_diagnostic,
        momentum=self.conf.momentum,
        weight_decay=self.conf.weight_decay,
        init_weights=self.init_weights)

    tf.logging.info("Done.")

    return model

  def train(self, train_dataset, post_create_model_hook=None):
    """Trains a model given a dataset provider.

    Args:
      train_dataset: Dataset provider.
      post_create_model_hook: Callback called after model creation,
        but before training.

    Yields:
      Epoch index and model tuple at the end of every pass.
    """
    g = tf.Graph()
    with g.as_default():
      session = tf.Session(graph=g)
      with session.as_default():
        model = self.create_model(
            input_size=train_dataset.get_input_dim(),
            output_size=train_dataset.get_output_dim(),
            dataset_size=train_dataset.size)

        if post_create_model_hook:
          post_create_model_hook(model)

        tf.logging.info("Initializing global variables...")
        session.run(tf.global_variables_initializer())
        tf.logging.info("Done.")

        tf.logging.info("Snapshot before training.")
        yield 0, model

        for p in range(1, self.conf.passes + 1):

          train_dataset.reset_and_reshuffle(p)
          model.do_pass(train_dataset, self.conf.batch_size)

          do_snapshot = False
          if self.conf.log2_snapshots:
            do_snapshot = is_power2(p)
          else:
            do_snapshot = (p % self.conf.snapshot_every_n_passes) == 0

          if do_snapshot:
            tf.logging.info("Snapshot after pass = %d", p)
            yield p, model

    tf.reset_default_graph()

  def get_filename_path(self):
    """Returns results filename path."""
    path = os.path.join(self.conf.result_path,
                        self.conf.result_filename % self.conf.__dict__)
    return path

  def exists(self):
    if self.is_persistent_experiment:
      path = self.get_filename_path()
      file_exists = tf.gfile.Exists(path)
      if not file_exists:
        tf.logging.info("Will save results to \"%s\".", path)
      return file_exists

  def save(self, rec):
    if self.is_persistent_experiment:
      path = self.get_filename_path()
      extended_rec = copy(rec)

      for (k, v) in self.conf.__dict__.items():
        if k not in extended_rec:
          extended_rec[k] = v

      tf.logging.info("Saving result to \"%s\"...", path)

      with tf.gfile.Open(path, "w") as gf:
        pkl.dump(extended_rec, gf)

      tf.logging.info("Done.")

  def load(self):
    path = self.get_filename_path()
    tf.logging.info("Loading result at \"%s\"...", path)
    with tf.gfile.Open(path, "r") as gf:
      return pkl.load(gf)

  def get_data(self):
    """Retrieves and returns the whole dataset according to experiment spec.

    Returns:
      Dataset provider.
    """
    tf.logging.info(
        "Retrieving data for the \"%s\", split = %d, split size = %d",
        self.conf.dataset, self.conf.split_i, self.conf.split_n)
    data = get_data(
        self.conf.dataset,
        self.conf.data_path,
        self.conf.split_i,
        self.conf.split_n,
        flip_label_ratio=self.conf.flip_labels)
    tf.logging.info("Done.")
    return data
