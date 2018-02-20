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
from dl_bounds.src.exp_helpers import sample_n_loo_indices
from dl_bounds.src.experiments.exp_base import Experiment
import numpy as np
import tensorflow as tf


class LOOExperiment(Experiment):
  """Leave-one-out experiment.

  Evaluates first and second order subsampled leave-one-out error
  and stores corresponding weights.
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

    # LOO index combinations
    loo_ix, _, _ = sample_n_loo_indices(self.conf.data_shuffle_seed,
                                        self.conf.loo_sampling_n, 0, m)
    n_loo_ix = len(loo_ix)
    loo_i_indices, loo_j_indices = zip(*loo_ix)

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)

    n_records = self.get_n_records()

    loo_losses = np.zeros((n_records, n_loo_ix))
    lto_losses = np.zeros((n_records, n_loo_ix))
    loo2_loss_prod = np.zeros((n_records, n_loo_ix))
    losses_on_loo_i = np.zeros((n_records, n_loo_ix))
    losses_on_loo_j = np.zeros((n_records, n_loo_ix))
    weight_l2_dist = np.zeros((n_records, n_loo_ix))
    weights = []

    tf.logging.info("Computing Rval(A_S) and Remp(A_S)...")

    for (pass_index, (p, model)) in enumerate(self.train(train_dataset)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)

      # Saving weights to compare against LOO weights
      weights.append(model.weights.eval())

      # Computing losses on the LOO points, this is needed
      # for sample variance computation
      losses_on_loo_i[pass_index, :] = model.losses_on_sample(
          x_train[loo_i_indices, :],
          y_train[loo_i_indices, :])

      losses_on_loo_j[pass_index, :] = model.losses_on_sample(
          x_train[loo_j_indices, :],
          y_train[loo_j_indices, :])

    tf.logging.info("Computing Rloo(A, S)")

    for (loo_index, (i, _)) in enumerate(loo_ix):
      tf.logging.info("First-order LOO: %d/%d", loo_index + 1, n_loo_ix)
      y_train_ = copy(y_train)
      y_train_[i, :] = -1

      loo_train_dataset = LocalDatasetProvider(
          x_train, y_train_, shuffle_seed=self.conf.data_shuffle_seed)

      for (pass_index, (p, model)) in enumerate(self.train(loo_train_dataset)):
        loo_losses[pass_index, loo_index] = model.mean_loss_on_sample(
            x_train[[i], :], y_train[[i], :])

        model_weights = model.weights.eval()
        weight_l2_dist[pass_index, loo_index] = np.linalg.norm(
            weights[pass_index] - model_weights)

        # tf.logging.info("Passes: %s", ", ".join(
        #   map(str, range(0, p + 1, self.conf.snapshot_every_n_passes))))
        tf.logging.info("LOO: %s", ", ".join(
            ["%.2f" % x for x in
             loo_losses[:pass_index + 1, :loo_index + 1].mean(axis=1)]
        ))

    if self.conf.compute_second_order_loo:
      tf.logging.info("Computing second-order LOO interactions")

      for (loo_index, (i, j)) in enumerate(loo_ix):
        tf.logging.info("Second-order LOO: %d/%d", loo_index + 1, n_loo_ix)

        y_train_ = copy(y_train)
        y_train_[i, :] = -1
        y_train_[j, :] = -1

        loo_train_dataset = LocalDatasetProvider(
            x_train, y_train_, shuffle_seed=self.conf.data_shuffle_seed)

        for (pass_index, (p, model)) in enumerate(
            self.train(loo_train_dataset)):
          loo_ij_loss_i = model.mean_loss_on_sample(x_train[[i], :],
                                                    y_train[[i], :])
          loo_ij_loss_j = model.mean_loss_on_sample(x_train[[j], :],
                                                    y_train[[j], :])

          loo2_loss_prod[pass_index, loo_index] = loo_ij_loss_i * loo_ij_loss_j

    results = self.get_train_val_measurements()
    results.update(
        dict(
            loo_losses=loo_losses,
            lto_losses=lto_losses,
            losses_on_loo_i=losses_on_loo_i,
            losses_on_loo_j=losses_on_loo_j,
            loo2_loss_prod=loo2_loss_prod,
            loo_indices=loo_ix,
            weight_l2_dist=weight_l2_dist))

    self.save(results)

    return results
