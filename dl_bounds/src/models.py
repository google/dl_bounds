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

"""Implements base Model class, basic CNNs and MLP primitives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from copy import copy
from dl_bounds.src.metrics import compute_spectral_complexity
from dl_bounds.src.metrics import emp_entropy
from dl_bounds.src.metrics import PathL2Norm
import numpy as np
from scipy.spatial.distance import pdist
import tensorflow as tf


class Model(object):
  """Simple base class for neural nets.

  Implements base class for convolutional or fully-connected network that
  supports various complexity measures. All weights are stored internally
  as a vector which permits easier computation of some complexity measures.
  """

  def __init__(self):
    self.reset()

  def reset(self):
    """Resets the model."""
    self.net_in = None
    self.net_out = None
    self.label_placeholder = None
    self.get_weights = None
    self.pooling_divisor = 1.0
    self.input_shape = None
    self.weight_cursor = 0

    self.layer_coords = []
    self.layer_inputs = []

    self.pflug_diagnostic = False
    self.pflug_diagnostic_track = []
    self.pflug_cos_diagnostic_track = []

    self.weight_decay_op = None

  def add_input(self, input_shape):
    self.input_shape = input_shape
    self.net_in = tf.placeholder(tf.float32, shape=[None, np.prod(input_shape)])
    with tf.name_scope("input_reshape"):
      self.net_out = tf.reshape(self.net_in, [-1] + input_shape)

    self.layer_inputs = [self.net_in]

  def add_output(self, num_outputs):
    self.label_placeholder = tf.placeholder(
        tf.float32, shape=[None, num_outputs])
    self.add_fc("output", num_outputs)

    self.layer_inputs.append(self.net_out)

  def add_conv2d(self, name, ksize, kernels, stride=1):
    with tf.name_scope(name):
      chan = int(self.net_out.shape[-1])
      shape = [ksize, ksize, chan, kernels]

      weights_conv = self.get_weights(shape)
      b_conv = self.get_weights([shape[-1]])
      h_conv = tf.nn.conv2d(
          self.net_out, weights_conv, strides=[1, stride, stride, 1],
          padding="SAME")
      self.net_out = h_conv + b_conv

  def add_pool(self, name, size, stride=2):
    with tf.name_scope(name):
      self.net_out = tf.nn.max_pool(
          self.net_out,
          ksize=[1, size, size, 1],
          strides=[1, stride, stride, 1],
          padding="SAME")
      self.pooling_divisor *= stride

  def add_relu(self):
    self.net_out = tf.nn.relu(self.net_out)

    self.layer_inputs.append(self.net_out)

  def flatten(self):
    prev_num_kernels = int(self.net_out.shape[-1])
    shrunk_input_width = int(self.input_shape[0] / self.pooling_divisor)
    shrunk_input_height = int(self.input_shape[1] / self.pooling_divisor)

    self.net_out = tf.reshape(self.net_out, [-1, (shrunk_input_width *
                                                  shrunk_input_height *
                                                  prev_num_kernels)])

  def add_fc(self, name, size):
    with tf.name_scope(name):
      prev_dim = int(self.net_out.shape[-1])

      self.layer_coords.append((self.weight_cursor, prev_dim, size))

      weights_fc = self.get_weights([prev_dim, size])
      b_fc = self.get_weights([size])
      self.net_out = tf.matmul(self.net_out, weights_fc) + b_fc

  def get_fc_weights(self):
    weights = self.weights.eval()

    layers = []

    for (cursor, d1, d2) in self.layer_coords:
      layer_weights = weights[cursor:cursor + d1 * d2]
      layers.append(layer_weights.reshape((d1, d2)))

    return layers

  def path_l2_norm(self):
    layers = self.get_fc_weights()
    norm = PathL2Norm()
    return norm(layers)

  def spectral_complexity(self):
    layers = self.get_fc_weights()
    return compute_spectral_complexity(layers)

  def prod_of_frob_layers(self):
    layers = self.get_fc_weights()
    return np.prod([np.linalg.norm(weights, "fro") for weights in layers])

  def compute_layer_distortions(self, instances, y, metric="euclidean"):
    """Computes layer distortions measures.

    Treating each layer as a non-linear embedding, compute their distortions.

    Args:
      instances: examples to evaluate distortion on.
      y: corresponding labels.
      metric: type of metric used in distribution.

    Returns:
      dictionary with measurements:
        mean_shallow_distortion: Mean distortions of every layer.
        std_shallow_distortion: Corresponding standard deviations.
        mean_deep_distortion: Mean distortions of every cascade of layers
                              (e.g. f1(.), f1(f2(.)), etc)
        std_deep_distortion: Corresponding standard deviations.
        ratio_of_mean_shallow_embeddings: Pseudo-distortion, where one takes
          ratio of means, rather than mean of ratios.
        ratio_of_mean_deep_embeddings: Corresponding cascade version.
    """
    feed_dict = {self.net_in: instances, self.label_placeholder: y}

    x_dists = pdist(instances, metric=metric)
    x_mean = np.mean(x_dists)

    results = dict(
        ratio_of_mean_shallow_embeddings=[],
        ratio_of_mean_deep_embeddings=[],
        mean_shallow_distortion=[],
        std_shallow_distortion=[],
        mean_deep_distortion=[],
        std_deep_distortion=[])

    for i in range(len(self.layer_inputs) - 1):
      embedding_inputs = self.layer_inputs[i].eval(feed_dict=feed_dict)
      embedding = self.layer_inputs[i + 1].eval(feed_dict=feed_dict)

      embedding_dists = pdist(embedding, metric=metric)
      embedding_mean = np.mean(embedding_dists)

      embedding_inputs_dists = pdist(embedding_inputs, metric=metric)
      embedding_inputs_mean = np.mean(embedding_inputs_dists)

      shallow_distortions = embedding_dists / embedding_inputs_dists
      deep_distortions = embedding_dists / x_dists

      results["ratio_of_mean_shallow_embeddings"].append(
          embedding_mean / embedding_inputs_mean)
      results["ratio_of_mean_deep_embeddings"].append(embedding_mean / x_mean)

      results["mean_shallow_distortion"].append(np.mean(shallow_distortions))
      results["std_shallow_distortion"].append(np.std(shallow_distortions))

      results["mean_deep_distortion"].append(np.mean(deep_distortions))
      results["std_deep_distortion"].append(np.std(deep_distortions))

    return results

  def sharpness(self, dataset, batch_size, learning_rate, init_stddev, passes,
                optimizer, alpha):
    """Computes sharpness of the network as described in N.

    Sharpness is described in:
    S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang.
    On large-batch training for deep learning: Generalization gap and
    sharp minima. In ICLR 2017.

    Args:
      dataset: dataset provider.
      batch_size: batch size.
      learning_rate: learning rate.
      init_stddev: stddev of weight initialization distribution (normal dist.).
      passes: number of epochs.
      optimizer: type of optimizer as required by Model.initialize.
      alpha: hyperparameter of sharpness.

    Returns:
      value of sharpness.
    """

    model = copy(self)

    memo_weights = self.weights.eval()
    abs_memo_weights = np.abs(memo_weights)

    w = tf.Variable(memo_weights)
    v = tf.Variable(tf.zeros(len(memo_weights)))

    def _allocate_weights(_):
      model.weights = w + v
      model.get_weights = model.get_weight_chunk
      model.weight_cursor = 0

    model.allocate_weights = _allocate_weights
    model.initialize(
        init_stddev=init_stddev,
        learning_rate=learning_rate,
        seed=1,
        optimizer=optimizer,
        pflug_diagnostic=False)

    tf.get_default_session().run([w.initializer, v.initializer])

    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss=tf.negative(model.mean_loss), var_list=[v])

    if dataset.size < batch_size:
      batch_size = dataset.size

    for i_pass in range(passes):
      dataset.reset_and_reshuffle(i_pass)

      for _ in range(int(dataset.size / batch_size)):
        x_mb, y_mb = dataset.read_next(batch_size)
        feed_dict = {model.net_in: x_mb, model.label_placeholder: y_mb}

        opt.run(feed_dict=feed_dict)

        # Coordinate-wise projection step
        v_np = v.eval()
        trunc_ix = np.abs(v_np) > alpha * (abs_memo_weights + 1)
        if trunc_ix.any():
          v_np[trunc_ix] = alpha * (memo_weights[trunc_ix] + 1)
          v.load(v_np)

    emp_risk_adv = model.mean_loss_on_dataset(dataset)
    emp_risk = self.mean_loss_on_dataset(dataset)
    sharpness = (emp_risk_adv - emp_risk) / (1.0 + emp_risk)

    return sharpness

  def weight_variance(self):
    weights = self.weights.eval()
    return np.var(weights)

  def weight_entropy(self):
    weights = self.weights.eval()
    return emp_entropy(weights - weights.min())

  def build(self):
    raise NotImplementedError

  def count_parameters(self):
    """Counts number of parameters in the model.

    Returns:
      number of parameters.
    """
    def _get_dummy_weights(shape):
      return tf.Variable(tf.zeros(shape))

    get_weights = self.get_weights
    self.get_weights = _get_dummy_weights

    # Building dummy graph to do parameter counting
    g = tf.Graph()
    with g.as_default():
      self.build()
      self.reset()

      total_params = 0
      for var in tf.trainable_variables():
        total_params += int(np.prod(var.shape))

    self.get_weights = get_weights
    return total_params

  def get_weight_chunk(self, chunk_shape):
    n = np.prod(chunk_shape)

    if (self.weight_cursor + n) <= self.weights.get_shape()[0]:
      w = tf.reshape(
          tf.slice(self.weights, begin=[self.weight_cursor], size=[n]),
          chunk_shape)
      self.weight_cursor += n

      return w
    else:
      raise Exception("Used up all allocated weights.")

  def allocate_weights(self, init_stddev):
    self.num_params = self.count_parameters()

    tf.logging.info("Allocating weights [%s] ~ truncated_normal(stddev=%s)",
                    self.num_params, init_stddev)

    init = tf.truncated_normal([self.num_params], stddev=init_stddev)
    self.weights = tf.Variable(init)
    self.get_weights = self.get_weight_chunk
    self.weight_cursor = 0

  def allocate_weights_from_array(self, array):
    self.num_params = self.count_parameters()

    tf.logging.info("Allocating weights [%s] provided array", self.num_params)

    init = tf.constant(array)
    self.weights = tf.Variable(init)
    self.get_weights = self.get_weight_chunk
    self.weight_cursor = 0

  def create_losses(self):
    """Creates loss functions."""
    tf.logging.info("Creating mean cross entropy loss")

    self.mean_loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label_placeholder, logits=self.net_out))

    correct_pred = tf.equal(
        tf.argmax(self.net_out, 1), tf.argmax(self.label_placeholder, 1))
    self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    self.multi_loss = tf.losses.softmax_cross_entropy(
        onehot_labels=self.label_placeholder,
        logits=self.net_out,
        reduction=tf.losses.Reduction.NONE)

  def create_optimizer(self,
                       learning_rate,
                       name="sgd",
                       momentum=0,
                       weight_decay=0):
    """Creates optimizer for the model.

    Args:
      learning_rate: learning rate.
      name: type of optimizer. Supported: sgd, adam.
      momentum: Momentum.
      weight_decay: Weight decay.

    """
    if name == "sgd":
      if momentum == 0:
        tf.logging.info("Creating SGD optimizer, lr = %s", learning_rate)
        self.opt = tf.train.GradientDescentOptimizer(learning_rate)
      else:
        tf.logging.info("Creating SGD optimizer, lr = %s, momentum = %s",
                        learning_rate, momentum)
        self.opt = tf.train.MomentumOptimizer(learning_rate, momentum)

    elif name == "adam":
      tf.logging.info("Creating Adam optimizer, lr = %s", learning_rate)
      self.opt = tf.train.AdamOptimizer(learning_rate)

    total_loss = self.mean_loss

    if weight_decay > 0:
      tf.logging.info("Using weight decay = %s", weight_decay)
      self.weight_decay_op = tf.multiply(
          tf.nn.l2_loss(self.weights), weight_decay, name="L2_regularizer")
      total_loss += self.weight_decay_op

    self.minimizer = self.opt.minimize(loss=total_loss)
    self.grad = tf.gradients(total_loss, self.weights)[0]

  def initialize(self,
                 init_stddev,
                 learning_rate,
                 seed,
                 optimizer,
                 pflug_diagnostic,
                 momentum=0,
                 weight_decay=0,
                 init_weights=None):
    """Initializes the model.

    Args:
      init_stddev: stddev of initialization distribution.
      learning_rate: learning rate.
      seed: seed of the model. Specifies the initilization position.
      optimizer: name of the optimizer (see create_optimizer).
      pflug_diagnostic: whether to record Pflug statistics.
      momentum:
      weight_decay:
      init_weights: initialize weights from init_weights.
    """
    tf.set_random_seed(seed)

    if init_weights is not None:
      self.allocate_weights_from_array(init_weights)
    else:
      self.allocate_weights(init_stddev)
    self.build()

    assert self.net_out is not None
    assert self.label_placeholder is not None
    assert self.weight_cursor == self.num_params

    self.create_losses()
    self.create_optimizer(
        learning_rate,
        name=optimizer,
        momentum=momentum,
        weight_decay=weight_decay)

    self.pflug_diagnostic = pflug_diagnostic

  def mean_loss_on_sample(self, instances, y):
    feed_dict = {self.net_in: instances, self.label_placeholder: y}
    return self.mean_loss.eval(feed_dict=feed_dict)

  def losses_on_sample(self, instances, y):
    feed_dict = {self.net_in: instances, self.label_placeholder: y}
    return self.multi_loss.eval(feed_dict=feed_dict)

  def mean_metric_on_dataset(self, dataset_provider_, metric):
    """Computer mean metric on a given dataset.

    Args:
      dataset_provider_: dataset provider.
      metric: tensorflow graph with scalar output.

    Returns:
      mean of the metric on a given dataset.
    """
    dataset_provider = copy(dataset_provider_)
    dataset_provider.reset_and_reshuffle(1)

    total_metric = 0.0

    n = 0
    while dataset_provider.has_more():
      x_mb, y_mb = dataset_provider.read_next(1000)
      size_mb = x_mb.shape[0]
      feed_dict = {self.net_in: x_mb, self.label_placeholder: y_mb}

      if total_metric is None:
        total_metric = metric.eval(feed_dict=feed_dict)
      else:
        total_metric = ((float(n) / (n + size_mb))
                        * total_metric
                        + (float(size_mb) / (n + size_mb))
                        * metric.eval(feed_dict=feed_dict))

      n += size_mb

    return total_metric

  def mean_loss_on_dataset(self, dataset_provider_):
    return self.mean_metric_on_dataset(dataset_provider_, self.mean_loss)

  def accuracy_on_dataset(self, dataset_provider_):
    return self.mean_metric_on_dataset(dataset_provider_, self.accuracy)

  def eps_margin(self, dataset_provider_, eps_range):
    """Computes sofr epsilon-margin.

    Described in:
    L. Bartlett, D. J. Foster, and M. J. Telgarsky. Spectrally-normalized
    margin bounds for neural networks. In NIPS 2017.

    Args:
      dataset_provider_: dataset provider.
      eps_range: list of margin softness.

    Returns:
      array of margins.
    """

    dataset_provider = copy(dataset_provider_)
    dataset_provider.reset_and_reshuffle(1)

    margins = []

    n = 0
    while dataset_provider.has_more():
      x_mb, y_mb = dataset_provider.read_next(1000)
      size_mb = x_mb.shape[0]
      feed_dict = {self.net_in: x_mb, self.label_placeholder: y_mb}

      outs = self.net_out.eval(feed_dict=feed_dict)
      margins_mb = [
          outs[i, y_mb[i, :] == 1] - outs[i, y_mb[i, :] == 0].max()
          for i in range(1000)
      ]
      margins.append(margins_mb)

      n += size_mb

    margins = np.vstack(margins)
    margins.sort(axis=0)
    soft_margins = np.array(
        [margins[int(eps * margins.size)][0] for eps in eps_range])
    return soft_margins

  def do_pass(self, train_dataset_provider, batch_size):
    """Performs one pass (epoch) over the data.

    Args:
      train_dataset_provider: provider for the training data.
      batch_size: batch size.

    Returns:
      number of consumed examples.
    """
    consumed_examples = 0

    if train_dataset_provider.size < batch_size:
      batch_size = train_dataset_provider.size

    for _ in range(int(train_dataset_provider.size / batch_size)):
      x_mb, y_mb = train_dataset_provider.read_next(batch_size)
      feed_dict = {self.net_in: x_mb, self.label_placeholder: y_mb}

      if self.pflug_diagnostic:
        gradients_t = self.compute_gradient(x_mb, y_mb)

      self.minimizer.run(feed_dict=feed_dict)
      consumed_examples += x_mb.shape[0]

      if self.pflug_diagnostic:
        gradients_t_plus_one = self.compute_gradient(x_mb, y_mb)
        self.pflug_diagnostic_track.append(
            gradients_t_plus_one.dot(gradients_t))

        normalizer = np.linalg.norm(gradients_t_plus_one) * np.linalg.norm(
            gradients_t)
        self.pflug_cos_diagnostic_track.append(
            gradients_t_plus_one.dot(gradients_t) / normalizer)

    return consumed_examples

  def compute_gradient(self, instances, y):
    feed_dict = {self.net_in: instances, self.label_placeholder: y}
    return self.grad.eval(feed_dict=feed_dict)

  def compute_gradient_on_dataset(self, dataset):
    return self.mean_metric_on_dataset(dataset, metric=self.grad)


class MnistCnn(Model):

  def build(self):
    self.add_input([28, 28, 1])

    self.add_conv2d("conv1", ksize=5, kernels=32)
    self.add_relu()
    self.add_pool("pool1", size=2)

    self.add_conv2d("conv2", ksize=5, kernels=64)
    self.add_relu()
    self.add_pool("pool2", size=2)

    self.flatten()

    self.add_fc("fc1", size=1024)
    self.add_relu()

    self.add_output(10)


class Cifar10Cnn(Model):

  def build(self):
    self.add_input([32, 32, 3])

    self.add_conv2d("conv1", ksize=5, kernels=64)
    self.add_relu()
    self.add_pool("pool1", size=3)

    self.add_conv2d("conv2", ksize=5, kernels=64)
    self.add_relu()
    self.add_pool("pool2", size=3)

    self.flatten()

    self.add_fc("fc1", size=384)
    self.add_relu()

    self.add_fc("fc2", size=192)
    self.add_relu()

    self.add_output(10)


class MLPCustom(Model):

  def __init__(self, in_size, out_size, hidden_sizes):
    super(MLPCustom, self).__init__()
    self.in_size = in_size
    self.out_size = out_size
    self.hidden_sizes = hidden_sizes

  def build(self):

    self.add_input([self.in_size])

    for (i, hl_size) in enumerate(self.hidden_sizes):
      self.add_fc("fc%d" % i, hl_size)
      self.add_relu()

    self.add_output(self.out_size)


ALL_MODELS = {("mnist", "cnn"): MnistCnn, ("cifar10", "cnn"): Cifar10Cnn}
