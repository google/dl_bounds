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

"""Implements utility experiments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import copy
from itertools import product
import math
from dl_bounds.src.data import LocalDatasetProvider
from dl_bounds.src.exp_helpers import aggregate_dicts
from dl_bounds.src.exp_helpers import is_power2
from dl_bounds.src.experiments.exp_base import Experiment
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


class ModelSelectionExperiment(Experiment):
  """Meta-experiment for model selection."""

  @staticmethod
  def get_grid_as_dict():
    return dict(
        sgd=dict(
            learning_rate=[1e-1, 1e-2, 1e-3],
            batch_size=[64, 128],
            init_stddev=[1e-1, 1e-2],
            exp_decay=[True, False]),
        adam=dict(
            learning_rate=[1e-2, 1e-3, 1e-4],
            batch_size=[64, 128],
            init_stddev=[1e-1, 1e-2],
            exp_decay=[True, False]))

  @staticmethod
  def get_flat_grid():
    grid = ModelSelectionExperiment.get_grid_as_dict()
    flat_grid = []
    for (opt_alg, hp) in grid.items():
      for p in product(*hp.values()):
        grid_point = dict(zip(hp.keys(), p))
        grid_point["optimizer"] = opt_alg
        flat_grid.append(grid_point)

    return flat_grid

  @staticmethod
  def get_grid_card():
    flat_grid = ModelSelectionExperiment.get_flat_grid()
    return len(flat_grid)

  def run(self):
    """Runs experiment."""
    base_conf = copy(self.conf)

    flat_grid = ModelSelectionExperiment.get_flat_grid()
    hps = flat_grid[base_conf.grid_i]
    base_conf.__dict__.update(hps)
    base_conf.snapshot_every_n_passes = base_conf.passes

    exp = TrainEvalExperiment(base_conf)
    rs = exp.run()

    rs.update(hps)

    self.save(rs)
    return rs


class TrainEvalExperiment(Experiment):
  """Simple experiment to evaluate metric described in the base class.

     Used for model selection.
  """

  def run(self):
    """Runs experiment."""
    (x_train, y_train, _, _, _) = self.get_data()

    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=self.conf.data_shuffle_seed)

    m = x_train.shape[0]
    tf.logging.info("Training set size = %d", m)
    tf.logging.info("Val set size = %d", x_val.shape[0])

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)

    for (pass_index, (p, model)) in enumerate(self.train(train_dataset)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)

    results = self.get_train_val_measurements()

    return results


class MultisplitExperiment(Experiment):

  def __init__(self, conf, subexp_factory):
    conf.snapshot_every_n_passes = conf.passes
    conf.log2_snapshots = False

    super(MultisplitExperiment, self).__init__(conf)
    self.subexp_factory = subexp_factory

  def run(self):
    """Runs multisplit experiment.

       Runs multiple experiments on log2 split progression up to
       split_n (must be power of 2).
    """
    if self.exists():
      tf.logging.info("Result file already exists.")
      return

    assert is_power2(self.conf.split_n)

    all_rs = []

    split_progression = 2**np.arange(4, int(math.log(self.conf.split_n, 2)) + 1)
    for n in split_progression:
      tf.logging.info("===========================================")
      tf.logging.info("MultisplitExperiment:: n=%d", n)
      tf.logging.info("===========================================")
      conf_ = copy(self.conf)
      conf_.split_n = n
      conf_.split_i = 0
      exp = self.subexp_factory(conf_)
      exp.is_persistent_experiment = False

      rs = exp.run()
      all_rs.append(rs)

    aggregated_rs = aggregate_dicts(all_rs)

    self.save(aggregated_rs)
