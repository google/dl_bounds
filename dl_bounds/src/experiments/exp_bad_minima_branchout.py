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
from dl_bounds.src.data import LocalDatasetProvider
from dl_bounds.src.exp_helpers import aggregate_dicts
from dl_bounds.src.experiments.exp_base import Experiment
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf


class BadMinimaBranchoutExperiment(Experiment):
  """Runs the branchout version of "bad minima" experiment."""

  def __init__(self, conf, subexp_factory):
    super(BadMinimaBranchoutExperiment, self).__init__(conf)
    self.subexp_factory = subexp_factory

  def run(self):
    """Runs experiment."""
    tf.logging.info("Pre-training network with 50% labels flipped...")

    conf = copy(self.conf)
    conf.flip_labels = 0.5
    conf.split_n = -1
    conf.log2_snapshots = True
    exp = Experiment(conf)

    (x_train, y_train, _, _, _) = exp.get_data()
    noisy_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)

    all_rs = []
    bad_min_weight_snapshots = []

    # Training model on the dataset with 50% labels randomly flipped, while
    # keeping intermediate weights
    for (p, model) in exp.train(noisy_dataset):
      init_weights = model.weights.eval()
      bad_min_weight_snapshots.append(init_weights)

    # Training & evaluating models initialized from intermediate weights
    for (p, init_weights) in enumerate(bad_min_weight_snapshots):
      tf.logging.info(
          """Initializing weights and running actual experiment from
weights of noisy experiment at pass %d.""",
          p)
      exp = self.subexp_factory(self.conf)
      exp.is_persistent_experiment = False
      exp.init_weights = init_weights
      rs = exp.run()
      rs["bad_min_branchout_pass"] = p
      all_rs.append(rs)

    aggregated_rs = aggregate_dicts(all_rs)
    self.save(aggregated_rs)

    w_l2_norm_at_bad_min = np.linalg.norm(bad_min_weight_snapshots[-1])
    dim = len(bad_min_weight_snapshots[-1])

    new_init_w = truncnorm(
        a=-2 / self.conf.init_stddev,
        b=2 / self.conf.init_stddev,
        scale=self.conf.init_stddev).rvs(size=dim).astype(np.float32)

    new_init_w = (
        new_init_w / np.linalg.norm(new_init_w)) * w_l2_norm_at_bad_min

    conf = copy(self.conf)
    exp = self.subexp_factory(conf)
    exp.is_persistent_experiment = False
    exp.init_weights = new_init_w
    rs = exp.run()
    rs["blown_up_stddev"] = True

    self.conf.result_filename += "_blown_up_stddev"
    self.save(rs)

    return aggregated_rs
