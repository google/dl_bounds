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
from itertools import product
import math
import os
import random
from dl_bounds.src.data import get_data
from dl_bounds.src.data import LocalDatasetProvider
from dl_bounds.src.exp_helpers import aggregate_dicts
from dl_bounds.src.exp_helpers import is_power2
from dl_bounds.src.exp_helpers import sample_n_loo_indices
from dl_bounds.src.hessian_top_sv import HessianTopSV
from dl_bounds.src.models import ALL_MODELS
from dl_bounds.src.models import MLPCustom
import numpy as np
from scipy.stats import truncnorm
from sklearn.model_selection import train_test_split
import tensorflow as tf

from dl_bounds.src.experiments.exp_base import Experiment

class NormsExperiment(Experiment):
  """Trains model and computes various complexities (norms, etc.)."""

  def run(self):
    """Runs experiment."""
    if self.exists():
      tf.logging.info("Result file already exists.")
      return

    (x_train, y_train, x_val, y_val, _) = self.get_data()

    # Subsampling points for distortion evaluation
    x_distortion, _, y_distortion, _ = train_test_split(
        x_train,
        y_train,
        train_size=self.conf.stat_sample_n,
        random_state=self.conf.data_shuffle_seed)

    m = x_train.shape[0]
    tf.logging.info("Training set size = %d", m)
    tf.logging.info("Val set size = %d", x_val.shape[0])

    train_dataset = LocalDatasetProvider(
        x_train, y_train, shuffle_seed=self.conf.data_shuffle_seed)
    val_dataset = LocalDatasetProvider(
        x_val, y_val, shuffle_seed=self.conf.data_shuffle_seed)

    n_records = self.get_n_records()

    weight_l2_norms = np.zeros((n_records,))
    prod_weight_l2_norms = np.zeros((n_records,))
    path_l2_norms = np.zeros((n_records,))
    spectral_complexities = np.zeros((n_records,))
    spectral_products = np.zeros((n_records,))

    ratios_of_mean_shallow_embeddings = None
    ratios_of_mean_deep_embeddings = None
    std_deep_distortions = None
    mean_shallow_distortions = None
    std_shallow_distortions = None
    mean_deep_distortions = None
    cond_numbers = None

    tf.logging.info("Computing Rval(A_S), Remp(A_S), and estimating norms")

    for (pass_index, (p, model)) in enumerate(self.train(train_dataset)):
      self.measure_on_train_val(train_dataset, val_dataset, pass_index, p,
                                model)
      self.report_train_val(pass_index)

      # Computing layer embedding distortions
      #
      distortion_results = model.compute_layer_distortions(
          x_distortion, y_distortion)
      if ratios_of_mean_deep_embeddings is None:
        n_layers = len(distortion_results["ratio_of_mean_shallow_embeddings"])
        ratios_of_mean_shallow_embeddings = np.zeros((n_records, n_layers))
        ratios_of_mean_deep_embeddings = np.zeros((n_records, n_layers))
        mean_shallow_distortions = np.zeros((n_records, n_layers))
        mean_deep_distortions = np.zeros((n_records, n_layers))
        std_shallow_distortions = np.zeros((n_records, n_layers))
        std_deep_distortions = np.zeros((n_records, n_layers))
        cond_numbers = np.zeros((n_records, n_layers))

      ratios_of_mean_shallow_embeddings[pass_index, :] = distortion_results[
          "ratio_of_mean_shallow_embeddings"]
      ratios_of_mean_deep_embeddings[pass_index, :] = distortion_results[
          "ratio_of_mean_deep_embeddings"]
      mean_shallow_distortions[pass_index, :] = distortion_results[
          "mean_shallow_distortion"]
      mean_deep_distortions[pass_index, :] = distortion_results[
          "mean_deep_distortion"]
      std_shallow_distortions[pass_index, :] = distortion_results[
          "std_shallow_distortion"]
      std_deep_distortions[pass_index, :] = distortion_results[
          "std_deep_distortion"]

      weight_l2_norms[pass_index] = np.linalg.norm(model.weights.eval())
      prod_weight_l2_norms[pass_index] = model.prod_of_frob_layers()
      path_l2_norms[pass_index] = model.path_l2_norm()

      spec_rs = model.spectral_complexity()
      spectral_complexities[pass_index] = spec_rs["spectral_complexity"]
      spectral_products[pass_index] = spec_rs["spectral_product"]
      cond_numbers[pass_index, :] = spec_rs["cond_numbers"]

      tf.logging.info("Path L2 norm: %s", ", ".join(
          ["%.2f" % x for x in path_l2_norms[:pass_index + 1]]
      ))
      tf.logging.info("Spectral complexity: %s", ", ".join(
          ["%.2f" % x for x in spectral_complexities[:pass_index + 1]]
      ))
      tf.logging.info("Spectral product: %s", ", ".join(
          ["%.2f" % x for x in spectral_products[:pass_index + 1]]
      ))

    results = self.get_train_val_measurements()
    results.update(
        dict(
            weight_l2_norms=weight_l2_norms,
            prod_weight_l2_norms=prod_weight_l2_norms,
            path_l2_norms=path_l2_norms,
            spectral_complexities=spectral_complexities,
            spectral_products=spectral_products,
            cond_numbers=cond_numbers,
            ratios_of_mean_shallow_embeddings=ratios_of_mean_shallow_embeddings,
            ratios_of_mean_deep_embeddings=ratios_of_mean_deep_embeddings,
            std_deep_distortions=std_deep_distortions,
            mean_shallow_distortions=mean_shallow_distortions,
            std_shallow_distortions=std_shallow_distortions,
            mean_deep_distortions=mean_deep_distortions))

    self.save(results)

    return results
