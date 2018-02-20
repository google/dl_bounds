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
import random
from dl_bounds.src.data import LocalDatasetProvider
from dl_bounds.src.experiments.exp_base import Experiment
import numpy as np
import tensorflow as tf


class StabilityExperiment(Experiment):
  """Checks stability of the model.

  Trains the model and evaluates it on a
  slightly perturbed training sets (omitted points).
  """

  def run(self):
    """Runs experiment."""
    if self.exists():
      tf.logging.info("Result file already exists.")
      return

    (x_train, y_train, x_val, y_val, _) = self.get_data()

    m = x_train.shape[0]
    m_val = x_val.shape[0]
    tf.logging.info("Training set size = %d", m)
    tf.logging.info("Val set size = %d", m_val)

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)

    n_records = self.get_n_records()

    rand = random.Random(x=self.conf.data_shuffle_seed)
    loo_ix = range(m)
    loo_ix = rand.sample(loo_ix, self.conf.loo_sampling_n)
    mvt_eps = 0.01

    weight_l2_dist = np.zeros((n_records,))
    loo_indices = np.zeros((n_records,))
    segpoint_grad_norms = np.zeros((n_records, int(1.0 / mvt_eps) + 1))
    midpoint_grad_norms = np.zeros((n_records,))
    midpoint_approx_errors = np.zeros((n_records,))
    midpoint_alphas = np.zeros((n_records,))
    w_grad_norms = np.zeros((n_records,))
    wloo_grad_norms = np.zeros((n_records,))
    intact_weights = []
    f_as_z = np.zeros((n_records, len(loo_ix)))

    for (pass_index, (p, model)) in enumerate(self.train(train_dataset)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)
      intact_weights.append(model.weights.eval())

      f_as_z[pass_index, :] = model.losses_on_sample(x_train[loo_ix, :],
                                                     y_train[loo_ix, :])

    loo_index = self.conf.loo_index
    i = loo_ix[loo_index]
    tf.logging.info("Replace-one training & evaluation: %d/%d", loo_index + 1,
                    len(loo_ix))

    x_loo = x_train[i, :]
    y_loo = y_train[i, :]

    y_train_ = copy(y_train)
    y_train_[i, :] = -1

    loo_train_dataset = LocalDatasetProvider(
        x_train, y_train_, shuffle_seed=self.conf.data_shuffle_seed)

    for (pass_index, (p, model)) in enumerate(self.train(loo_train_dataset)):

      # Computing L2 distance between intact weights and generated from a
      # pertubed training set
      tf.logging.info("Computing ||A(S) - A(S\\i)||")
      weights = intact_weights[pass_index]
      loo_weights = model.weights.eval()

      weight_l2_dist[pass_index] = np.linalg.norm(weights - loo_weights)

      loo_index_after_shuffle = np.where(loo_train_dataset.index == i)[0][0]
      loo_indices[pass_index] = loo_index_after_shuffle

      # Computing norms of the gradients w.r.t. weights on the segment
      # between intract and pertubed weights
      tf.logging.info(
          """Computing ||nabla f((1-tau) A(S) + tau A(S^(i)))||
for %d settings of tau""",
          int(1.0 / mvt_eps))
      w_delta = weights - loo_weights
      best_midpoint_error = float("inf")
      best_midpoint = None
      best_midpoint_alpha = None
      best_midpoint_grad_norm = None

      for (alpha_i, alpha) in enumerate(np.arange(0.0, 1.0 + mvt_eps, mvt_eps)):
        w_segpoint = (1.0 - alpha) * loo_weights + alpha * weights

        # Computing gradient on the segment point
        model.weights.load(w_segpoint)
        grad_at_segpoint = model.compute_gradient([x_loo], [y_loo])
        segpoint_grad_norms[pass_index, alpha_i] = np.linalg.norm(
            grad_at_segpoint)

        # Checking if segment point is close to the midpoint
        f_asloo_z = model.losses_on_sample([x_loo], [y_loo])[0]
        f_delta = f_as_z[pass_index, loo_index] - f_asloo_z

        # ...exhaustive search among all the segment points
        midpoint_error = abs(grad_at_segpoint.dot(w_delta) - f_delta)
        if midpoint_error < best_midpoint_error:
          best_midpoint_error = midpoint_error
          best_midpoint = w_segpoint
          best_midpoint_alpha = alpha
          best_midpoint_grad_norm = segpoint_grad_norms[pass_index, alpha_i]

      # Restoring weights for consequent passes
      model.weights.load(loo_weights)

      midpoint_grad_norms[pass_index] = best_midpoint_grad_norm
      midpoint_approx_errors[pass_index] = best_midpoint_error
      midpoint_alphas[pass_index] = best_midpoint_alpha

      w_grad_norms[pass_index] = segpoint_grad_norms[pass_index, -1]
      wloo_grad_norms[pass_index] = segpoint_grad_norms[pass_index, 0]

      tf.logging.info("||nabla f(wloo)|| = %s", ",".join(
          map(str, wloo_grad_norms[:pass_index + 1])))

      tf.logging.info(
          """Found midpoint with ||nabla f(mid)|| = %s at alpha=%s,
and approx. error = %s""",
          best_midpoint_grad_norm, best_midpoint_alpha, best_midpoint_error)

    results = self.get_train_val_measurements()
    results.update(
        dict(
            weight_l2_dist=weight_l2_dist,
            loo_indices=loo_indices,
            segpoint_grad_norms=segpoint_grad_norms,
            midpoint_grad_norms=midpoint_grad_norms,
            midpoint_approx_errors=midpoint_approx_errors,
            midpoint_alphas=midpoint_alphas,
            w_grad_norms=w_grad_norms,
            wloo_grad_norms=wloo_grad_norms))

    self.save(results)

    return results
