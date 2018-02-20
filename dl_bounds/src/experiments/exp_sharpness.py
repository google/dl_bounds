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
from dl_bounds.src.data import LocalDatasetProvider
from dl_bounds.src.experiments.exp_base import Experiment
import numpy as np
import tensorflow as tf


class SharpnessExperiment(Experiment):
  """Computes sharpness complexity measure.

     Sharpness described in:
     N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy,
     and P. T. P. Tang. On large-batch training
     for deep learning: Generalization gap and sharp minima.
     In ICLR 2017.
  """

  def run(self):
    """Runs experiment."""
    if self.exists():
      tf.logging.info("Result file already exists.")
      return

    (x_train, y_train, x_val, y_val, _) = self.get_data()

    m = x_train.shape[0]
    tf.logging.info("Training set size = %d", m)
    tf.logging.info("Val set size = %d", x_val.shape[0])

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)
    sharpness_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)

    n_records = self.get_n_records()

    # Constants as in the paper describing sharpness measure
    alpha_range = [5e-4, 1e-3, 1e-2]

    n_alpha = len(alpha_range)
    sharpness = np.zeros((n_records, n_alpha))

    tf.logging.info("Computing sharpness on alpha=%s", ", ".join(
        map(str, alpha_range)))

    for (pass_index, (p, model)) in enumerate(self.train(train_dataset)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)

      for (i_alpha, alpha) in enumerate(alpha_range):
        sharpness[pass_index, i_alpha] = model.sharpness(
            sharpness_dataset,
            batch_size=self.conf.batch_size,
            learning_rate=self.conf.learning_rate,
            init_stddev=self.conf.init_stddev,
            passes=10,
            optimizer=self.conf.optimizer,
            alpha=alpha)

        tf.logging.info(
            "Sharpness (alpha=%s): %s", alpha,
            ", ".join(
                ["%.2f" % x for x in sharpness[:pass_index+1, i_alpha]]))

    results = self.get_train_val_measurements()
    results.update(dict(sharpness=sharpness, alpha=alpha_range))

    self.save(results)

    return results
