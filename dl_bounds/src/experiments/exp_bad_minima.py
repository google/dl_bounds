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
from dl_bounds.src.experiments.exp_base import Experiment
import tensorflow as tf


class BadMinimaExperiment(Experiment):
  """Runs "bad minima" experiment."""

  def __init__(self, conf, subexp_factory):
    super(BadMinimaExperiment, self).__init__(conf)
    self.subexp_factory = subexp_factory

  def run(self):
    """Runs experiment."""
    subexp = self.subexp_factory(self.conf)

    if subexp.exists():
      tf.logging.info("Result file already exists.")
      return

    tf.logging.info("Pre-training network with 50% labels flipped...")

    conf = copy(self.conf)
    conf.flip_labels = 0.5
    conf.split_n = -1
    conf.log2_snapshots = True
    exp = Experiment(conf)

    (x_train, y_train, _, _, _) = exp.get_data()
    noisy_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)

    # Training model on the dataset with 50% labels randomly flipped
    init_weights = None
    for (_, model) in exp.train(noisy_dataset):
      init_weights = model.weights.eval()

    # Training & evaluating model initialized from "bad" weights
    subexp.is_persistent_experiment = True
    subexp.init_weights = init_weights
    subexp.run()
