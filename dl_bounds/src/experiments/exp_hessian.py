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
from dl_bounds.src.hessian_top_sv import HessianTopSV
import numpy as np
import tensorflow as tf


class HessianExperiment(Experiment):
  """Hessian experiment.

  Trains the model, evaluates, and stores top singular
  value of the Hessian matrix.
  """

  def create_hessian_estimator(self, model):
    """Creates estimator for top singular value of the Hessian."""
    self.hessian_top_sv = HessianTopSV(model.mean_loss, model.weights,
                                       tf.logging.info)

  def run(self):
    """Runs experiment."""
    if self.exists():
      tf.logging.info("Result file already exists.")
      return

    (x_train, y_train, x_val, y_val, _) = self.get_data()

    m = x_train.shape[0]

    if self.conf.stat_sample_n == -1:
      x_hessian = x_train
      y_hessian = y_train
    else:
      tf.logging.info(
          "Subsampling training set for ||Hessian||_2 evaluation till %d.",
          self.conf.stat_sample_n)

      subset_ix = np.random.choice(
          np.arange(m), self.conf.stat_sample_n, replace=False)
      x_hessian = x_train[subset_ix, :]
      y_hessian = y_train[subset_ix]

    tf.logging.info("Training set size = %d", m)
    tf.logging.info("Val set size = %d", x_val.shape[0])
    tf.logging.info("Hessian set size = %d", x_hessian.shape[0])

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)

    n_records = self.get_n_records()

    hessian_top_sv_means = np.zeros((n_records,))
    hessian_top_sv_stds = np.zeros((n_records,))
    poweriter_means = np.zeros((n_records,))
    poweriter_stds = np.zeros((n_records,))

    tf.logging.info("""Computing Rval(A_S), Remp(A_S), and estimating
top Hessian singular values on the subsampled training sample...""")

    for (pass_index, (p, model)) in enumerate(
        self.train(train_dataset, self.create_hessian_estimator)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)

      hessian_rs = self.hessian_top_sv.compute(
          tf.get_default_session(),
          placeholder_x=model.net_in,
          placeholder_y=model.label_placeholder,
          data_x=x_hessian,
          data_y=y_hessian)

      hessian_top_sv_means[pass_index] = hessian_rs["avg_top_sv"]
      hessian_top_sv_stds[pass_index] = hessian_rs["std_top_sv"]
      poweriter_means[pass_index] = hessian_rs["avg_converge_steps"]
      poweriter_stds[pass_index] = hessian_rs["std_converge_steps"]

      tf.logging.info("E^[||H||_2]: %s", ", ".join(
          ["%.2f" % x for x in hessian_top_sv_means[:pass_index + 1]]
      ))

    results = self.get_train_val_measurements()
    results.update(
        dict(
            hessian_top_sv_means=hessian_top_sv_means,
            hessian_top_sv_stds=hessian_top_sv_stds,
            poweriter_means=poweriter_means,
            poweriter_stds=poweriter_stds))

    self.save(results)

    return results
