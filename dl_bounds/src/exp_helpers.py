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

"""Various tools for experimental code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import itertools as it
import random
import numpy as np


def is_power2(num):
  """States if a number is a power of two."""
  return num != 0 and ((num & (num - 1)) == 0)


def sample_n_loo_indices(seed, n, range_start, range_end):
  """Samples unique index pairs.

  Args:
    seed: random seed.
    n: number of pairs to sample.
    range_start: index range start.
    range_end: index range end.

  Returns:
    Tuple containing three lists:
    1) List of index pairs.
    2) Unique indices from the first position.
    3) Unique indices from the second position.

  """
  assert range_start <= range_end
  m = range_end - range_start

  assert int((m * (m - 1)) / 2) >= n

  rand = random.Random(seed)

  index_range = range(m)
  comb = it.combinations(index_range, 2)
  pick_indices = set(rand.sample(range(int(m * (m - 1) / 2)), n))
  ij_indices = [
      ij for (ij_index, ij) in enumerate(comb) if ij_index in pick_indices
  ]
  i_indices, j_indices = zip(*ij_indices)
  i_indices, j_indices = list(set(i_indices)), list(set(j_indices))

  return ij_indices, i_indices, j_indices


def sample_frac_loo_indices(seed, frac, range_start, range_end):
  m = range_end - range_start
  n = int(frac * int((m * (m - 1)) / 2))
  assert n > 0
  return sample_n_loo_indices(seed, n, range_start, range_end)


def get_split(instances, y, split_i, split_n):
  """Gets a random splits of an example set.

  Splits are done in a class-stratified way.

  Args:
    instances: example instances.
    y: example labels.
    split_i: split index.
    split_n: number of examples per split.

  Returns:
    Pair of instances and labels for a given split.
  """
  m = len(y)
  unique_labels = set(y)

  split_indices = []
  for l in unique_labels:
    class_indices = np.where(y == l)[0]

    n_per_class = (float(len(class_indices)) / m) * split_n
    split_start = int(split_i * n_per_class)
    split_end = int((split_i + 1) * n_per_class)
    split_indices.extend(class_indices[split_start:split_end])

  return instances[split_indices, :], y[split_indices]


def flip_labels(labels, ratio, seed=1):
  """Flips given proportion of labels.

  Args:
    labels: np.array with ordinal labels.
    ratio: probability of label flip.
    seed: random seed.

  Returns:
    np.array with flipped labels.

  Raises:
    ValueError: if ceil(ratio * num of labels) == 0
  """
  labels.setflags(write=1)
  n = len(labels)
  n_sample = int(ratio * n)
  if n_sample == 0:
    raise ValueError("Ratio is too small.")
  classes = list(set(labels))

  rand = random.Random(seed)
  index = np.arange(n)
  rand.shuffle(index)
  index = index[:n_sample]
  uni_labels = np.array([random.choice(classes) for _ in range(n_sample)])

  labels[index] = uni_labels

  return labels


def aggregate_dicts(all_rs):
  aggregated_rs = {}
  keys = all_rs[0].keys()
  for k in keys:
    aggregated_rs[k] = np.stack([rs_[k] for rs_ in all_rs]).squeeze()
  return aggregated_rs
