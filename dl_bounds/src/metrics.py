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

"""Implements various complexity measures for neural nets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import entropy


class PathL2Norm(object):
  """Implements Path L2-norm given MLP weight matrices.

  Path norm is computed through overloaded () operator.
  Described in B. Neyshabur, R. Tomioka, and N. Srebro.
  Norm-based capacity control in neural networks. In COLT 2015.
  """

  def __init__(self):
    self.cache = dict()

  def __call__(self, weights):
    """Computes the L2 Path norm.

    Args:
      weights: Weight matrices of MLP. Each matrix is of shape (in, out).

    Returns:
      float value of L2 Path norm.
    """
    outputs = weights[-1].shape[1]

    val = 0.0
    for i in range(outputs):
      val += self.compute_for_output_i(weights, len(weights) - 1, i)

    return val**0.5

  def compute_for_output_i(self, weights, depth, output_i):
    """Internal routine for path norm computation.

    Relies on dynamic programming over MLP graph.

    Args:
      weights: list of weight matrices of shape (in, out).
      depth: depth to compute for
      output_i: output index

    Returns:
      float component of L2 path norm for a particular output of MLP.
    """
    ins = weights[depth].shape[0]
    value = 0.0

    if depth > 0:
      for i in range(ins):
        key = (depth - 1, i)
        if key in self.cache:
          children_val = self.cache[key]
        else:
          children_val = self.compute_for_output_i(weights, depth - 1, i)
          self.cache[key] = children_val

        value += weights[depth][i, output_i]**2 * children_val
    else:
      for i in range(ins):
        value += weights[depth][i, output_i]**2

    return value


def compute_spectral_complexity(weights):
  """Implements measurement of spectral complexity.

  Spectral complexity of neural network is described in
  P. L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized
  margin bounds for neural networks. In NIPS 2017.

  Args:
    weights: Weight matrices of MLP. Each matrix is of shape (in, out).

  Returns:
    float value of the spectral complexity.
  """
  spectral_norms = []
  cond_numbers = []
  spectral_prod = 1.0
  factor = 0.0

  for layer_weights in weights:
    spec_layer_weights = np.linalg.norm(layer_weights, 2)
    spectral_norms.append(spec_layer_weights)
    spectral_prod *= spec_layer_weights

    cond_numbers.append(np.linalg.cond(layer_weights))

    unrolled_l1 = np.abs(layer_weights).sum()

    factor += (unrolled_l1 / spec_layer_weights)**(2.0 / 3.0)

  spectral_complexity = spectral_prod * factor**1.5

  return dict(
      spectral_product=spectral_prod,
      spectral_complexity=spectral_complexity,
      spectral_norms=spectral_norms,
      cond_numbers=cond_numbers)


def emp_entropy(x, bins=1000):
  """Computer empirical entropy of array x.

  Argument is first partitioned into the histogram.
  Then the entropy is computed over the histogram.

  Args:
    x: an array of floats.
    bins: number of bins in the histogram.

  Returns:
    entropy
  """
  if not np.isnan(x).any():
    return entropy(np.histogram(x, bins=bins)[0] / float(len(x)))
  else:
    return float("nan")
