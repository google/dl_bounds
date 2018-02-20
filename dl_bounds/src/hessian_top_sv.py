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

"""Implements Hessian top singular value estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
import tensorflow as tf


def is_power2(num):
  """States if a number is a power of two."""
  return num != 0 and ((num & (num - 1)) == 0)


def _hessian_vector_product(ys, xs, v):
  """Multiply the Hessian of `ys` wrt `xs` by `v`.

  This is an efficient construction that uses a backprop-like approach
  to compute the product between the Hessian and another vector. The
  Hessian is usually too large to be explicitly computed or even
  represented, but this method allows us to at least multiply by it
  for the same big-O cost as backprop.

  Implicit Hessian-vector products are the main practical, scalable way
  of using second derivatives with neural networks. They allow us to
  do things like construct Krylov subspaces and approximate conjugate
  gradient descent.

  Example: if `y` = 1/2 `x`^T A `x`, then `hessian_vector_product(y,
  x, v)` will return an expression that evaluates to the same values
  as (A + A.T) `v`.

  Args:
    ys: A scalar value, or a tensor or list of tensors to be summed to
        yield a scalar.
    xs: A list of tensors that we should construct the Hessian over.
    v: A list of tensors, with the same shapes as xs, that we want to
       multiply by the Hessian.

  Returns:
    A list of tensors (or if the list would be length 1, a single tensor)
    containing the product between the Hessian and `v`.

  Raises:
    ValueError: `xs` and `v` have different length.

  """

  # Validate the input
  length = len(xs)
  if len(v) != length:
    raise ValueError("xs and v must have the same length.")

  # First backprop
  grads = tf.gradients(ys, xs)

  assert len(grads) == length
  elemwise_products = [
      tf.multiply(grad_elem, tf.stop_gradient(v_elem))
      for grad_elem, v_elem in zip(grads, v)
      if grad_elem is not None
  ]

  # Second backprop
  return tf.gradients(elemwise_products, xs)


class HessianTopSV(object):
  """Implements measurement of top hessian singular value.

     Implemented through the power iteration method.
  """

  def __init__(self, fn, var, debug_logger=None):
    self.debug_logger = debug_logger

    v_init = None

    if len(var.shape) == 2:
      if int(var.shape[0]) == 1:
        v_shape = (int(var.shape[1]), 1)
      elif int(var.shape[1]) == 1:
        v_shape = (int(var.shape[0]), 1)
      else:
        raise ValueError("HessianTopSV:: variable has to be one-dimensional.")

      v = np.random.normal(size=v_shape).astype(np.float32)
      v_init = v / np.linalg.norm(v)
      v = tf.Variable(v_init)
    elif len(var.shape) == 1:
      d = int(var.shape[0])
      v = np.random.normal(size=(d,)).astype(np.float32)
      v_init = v / np.linalg.norm(v)
      v = tf.Variable(v_init)
    else:
      raise ValueError("HessianTopSV:: variable is of unsupported rank.")

    hv_prod = _hessian_vector_product(fn, [var], [v])[0]
    hv_prod_nmz = tf.nn.l2_normalize(hv_prod, 0)
    self.one_iter = tf.assign(v, hv_prod_nmz)
    self.hv_prod_norm = tf.norm(hv_prod)

    self.reset_v = v.assign(v_init)

  def compute(self,
              sess,
              placeholder_x,
              placeholder_y,
              data_x,
              data_y,
              max_iter=1000,
              eps=1e-2):
    """Computes top singular value of the Hessian.

    Args:
       sess: Tensorflow session.
       placeholder_x: Input placeholder.
       placeholder_y: Output placeholder.
       data_x: Data to compute the Hessian on.
       data_y: Corresponding labels.
       max_iter: Max. number of power iterations.
       eps: power iteration tolerance.

    Returns:
       A dictionary with fields:
         avg_top_sv: estimated top singular value.
         std_top_sv: 0 (backwards compat.)
         avg_converge_steps: number of power iterations taken to converge.
         std_converge_steps: 0 (backwards compat.)
    """

    norms = []
    steps = []
    feed_dict = {placeholder_x: data_x, placeholder_y: data_y}

    # Power iteration
    prev_norm = 0.0
    converge_steps = 0
    for power_iter in range(max_iter):
      sess.run(self.one_iter, feed_dict)

      converge_steps += 1

      if is_power2(power_iter):
        curr_norm = sess.run(self.hv_prod_norm, feed_dict)

        if abs(prev_norm - curr_norm) <= eps:
          break
        else:
          prev_norm = curr_norm

    curr_norm = sess.run(self.hv_prod_norm, feed_dict)
    sess.run(self.reset_v)

    if not math.isnan(curr_norm):
      norms.append(curr_norm)
      steps.append(converge_steps)

    self.debug_logger("SV = %.3f +/- %.1f, Iter = %.1f +/- "
                      "%.1f." % (np.mean(norms), np.std(norms),
                                 np.mean(steps), np.std(steps)))

    return dict(
        avg_top_sv=np.mean(norms),
        std_top_sv=np.std(norms),
        avg_converge_steps=np.mean(steps),
        std_converge_steps=np.std(steps))
