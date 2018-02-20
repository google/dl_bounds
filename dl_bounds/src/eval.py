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

"""Main evaluation file."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from argparse import Namespace
from dl_bounds.src.experiments.exp_bad_minima import BadMinimaExperiment
from dl_bounds.src.experiments.exp_hessian import HessianExperiment
from dl_bounds.src.experiments.exp_loo import LOOExperiment
from dl_bounds.src.experiments.exp_norms import NormsExperiment
from dl_bounds.src.experiments.exp_sharpness import SharpnessExperiment
from dl_bounds.src.experiments.exp_stability import StabilityExperiment
from dl_bounds.src.experiments.exp_util import ModelSelectionExperiment
from dl_bounds.src.experiments.exp_util import MultisplitExperiment
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("data_path", None, "Data path.")
tf.flags.DEFINE_string("result_path", None, "Result path.")
tf.flags.DEFINE_string("weight_path", None, "Weight path.")
tf.flags.DEFINE_string("result_filename", None,
                       "Result filename format string.")

tf.flags.DEFINE_enum("dataset", "mnist", ["mnist", "cifar10"],
                     "Dataset to train on.")
tf.flags.DEFINE_integer(
    "split_n", -1,
    "Size of a single dataset split (-1 stands for the whole dataset).")
tf.flags.DEFINE_integer("split_i", 0, "Split index.")

tf.flags.DEFINE_integer("data_shuffle_seed", 1,
                        "Random seed for data shuffle (e.g. for SGD pass).")

tf.flags.DEFINE_string("network", None, "Architecture name.")
tf.flags.DEFINE_integer("network_seed", 1,
                        "Random seed for network initialization.")
tf.flags.DEFINE_float("init_stddev", 0.01,
                      "Parameter of truncated normal initializer.")

tf.flags.DEFINE_integer(
    "snapshot_every_n_passes", 1,
    "Make experimental snapshot every <SNAPSHOT_EVERY_N_PASSES> pass(es).")

tf.flags.DEFINE_bool("log2_snapshots", False,
                     "Do snapshots at every the doubling pass.")

tf.flags.DEFINE_integer(
    "snapshot_every_n_mb", 10,
    """Make experimental snapshot every <SNAPSHOT_EVERY_N_MB>
mini-batches (experiment 'loo-single-pass' only).""")

tf.flags.DEFINE_integer("passes", 1, "Number of SGD passes.")
tf.flags.DEFINE_integer("batch_size", 50, "Batch size.")
tf.flags.DEFINE_string("optimizer", "sgd", "Optimizer type.")
tf.flags.DEFINE_float("learning_rate", 0.01, "Fixed SGD learning rate.")
tf.flags.DEFINE_float("momentum", 0, "Momentum.")
tf.flags.DEFINE_float("weight_decay", 0, "Weight decay.")

tf.flags.DEFINE_string("experiment", "norms", "Type of experiment.")
tf.flags.DEFINE_integer(
    "loo_sampling_n", 1000,
    "Number of leave-one-out evaluations (experiment=loo only).")
tf.flags.DEFINE_float(
    "flip_labels", 0,
    """Flip labels in the training set. The value is *100 percent
of labels to be flipped. Zero means not to flip labels.""")

tf.flags.DEFINE_bool(
    "collect_results", False,
    "Collect experimental results. Experiment will not be ran.")

tf.flags.DEFINE_bool("compute_second_order_loo", True,
                     "Compute second order LOO for variance estimation.")

tf.flags.DEFINE_integer("stat_sample_n", 1000,
                        """Sample size for computing various statistics
(e.g. hessian top singular value or distortion of layer embeddings).""")

tf.flags.DEFINE_bool("exp_decay", False, "Use exponential learning rate decay.")
tf.flags.DEFINE_integer(
    "num_passes_per_decay", 200,
    "Number of passes per one decay step (only for exp_decay=true).")
tf.flags.DEFINE_float(
    "learning_rate_decay_factor", 0.1,
    "Exponential learning rate decay factor (only for exp_decay=true).")

tf.flags.DEFINE_integer("grid_i", 0,
                        """Element of the model selection grid to evaluate at
(only for experiment=select_at).""")

tf.flags.DEFINE_bool("multisplit", False,
                     """Run multiple experiments on log2 split progression up to
split_n (must be power of 2).""")

tf.flags.DEFINE_integer("loo_index", 0,
                        "Leave-one-out index (only for experiment=stability).")

tf.flags.DEFINE_bool("pflug", False, """Do Pflug diagnostic of SGD
(i.e. keep track of grad(l(w_t+1))^T grad(l(w_t)) )""")

tf.flags.DEFINE_bool(
    "bad_min", False,
    """Initialize networks at the \"bad minimum\": weights are pre-trained
on a dataset with 50% labels flipped.""")

tf.flags.mark_flag_as_required("data_path")
tf.flags.mark_flag_as_required("result_path")
tf.flags.mark_flag_as_required("weight_path")
tf.flags.mark_flag_as_required("dataset")
tf.flags.mark_flag_as_required("network")


def main(unused_argv):
  conf = Namespace(
      data_path=FLAGS.data_path,
      weight_path=FLAGS.weight_path,
      result_path=FLAGS.result_path,
      result_filename=FLAGS.result_filename,
      dataset=FLAGS.dataset,
      split_n=FLAGS.split_n,
      split_i=FLAGS.split_i,
      data_shuffle_seed=FLAGS.data_shuffle_seed,
      network=FLAGS.network,
      network_seed=FLAGS.network_seed,
      init_stddev=FLAGS.init_stddev,
      snapshot_every_n_passes=FLAGS.snapshot_every_n_passes,
      snapshot_every_n_mb=FLAGS.snapshot_every_n_mb,
      log2_snapshots=FLAGS.log2_snapshots,
      passes=FLAGS.passes,
      batch_size=FLAGS.batch_size,
      learning_rate=FLAGS.learning_rate,
      momentum=FLAGS.momentum,
      weight_decay=FLAGS.weight_decay,
      optimizer=FLAGS.optimizer,
      experiment=FLAGS.experiment,
      loo_sampling_n=FLAGS.loo_sampling_n,
      flip_labels=FLAGS.flip_labels,
      compute_second_order_loo=FLAGS.compute_second_order_loo,
      stat_sample_n=FLAGS.stat_sample_n,
      exp_decay=FLAGS.exp_decay,
      num_passes_per_decay=FLAGS.num_passes_per_decay,
      learning_rate_decay_factor=FLAGS.learning_rate_decay_factor,
      grid_i=FLAGS.grid_i,
      multisplit=FLAGS.multisplit,
      loo_index=FLAGS.loo_index,
      pflug_diagnostic=FLAGS.pflug,
      bad_min=FLAGS.bad_min)

  exp_factories = {
      "loo": LOOExperiment,
      "hessian": HessianExperiment,
      "norms": NormsExperiment,
      "stability": StabilityExperiment,
      "select": ModelSelectionExperiment,
      "sharpness": SharpnessExperiment
  }

  tf.logging.set_verbosity(tf.logging.INFO)

  if conf.experiment == "select_size":
    tf.logging.info("Model selection grid cardinality: %s",
                    ModelSelectionExperiment.get_grid_card())
  else:
    exp_factory = exp_factories[conf.experiment]

    if conf.multisplit:
      exp = MultisplitExperiment(conf, exp_factory)
    elif conf.bad_min:
      exp = BadMinimaExperiment(conf, exp_factory)
    else:
      exp = exp_factory(conf)

    exp.run()


if __name__ == "__main__":
  tf.app.run(main)
