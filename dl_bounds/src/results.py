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

"""Implements visualization of exprimental results."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tempfile
from dl_bounds.src.pysqlite_ext import SqliteDB
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_enum("do", "plot_hessian_vs_margin",
                     ["plot_hessian_vs_margin", "table_norms", "table_phased"],
                     "Type of result to plot.")
tf.flags.DEFINE_string("db", None, "Database location.")
tf.flags.DEFINE_string("dataset", None, "Dataset.")
tf.flags.DEFINE_string("network", None, "Network.")
tf.flags.DEFINE_string("groupby_param", None, "Parameter name.")
tf.flags.DEFINE_string("pdf", None, "PDF filename to plot to.")
tf.flags.DEFINE_boolean("show", False, "Show plot.")

tf.flags.mark_flag_as_required("do")
tf.flags.mark_flag_as_required("db")
tf.flags.mark_flag_as_required("dataset")
tf.flags.mark_flag_as_required("network")
tf.flags.mark_flag_as_required("groupby_param")


class Results(object):
  """Retrieves results from Sqlite database."""

  def __init__(self, db_filename, mean_metrics=False):
    """Constructor.

    Args:
      db_filename: path to sqlite3 database.
      mean_metrics: return means of metrics over groupby parameter,
        passed to getters.

    """
    self.db_filename = db_filename

    self.common_metrics = [
        "train_error",
        "val_error",
        "train_zero_one_error",
        "val_zero_one_error",
    ]

    self.norm_metrics = [
        "weight_l2_norms", "path_l2_norms", "spectral_products",
        "spectral_complexities", "cond_numbers",
        "ratios_of_mean_deep_embeddings", "ratios_of_mean_shallow_embeddings",
        "soft_margins", "weight_variance", "weight_entropy",
        "train_grad_norm", "val_grad_norm",
    ]

    self.mean_metrics = mean_metrics

  @staticmethod
  def records_to_dict(rs, records, name):
    rs[name] = np.vstack([rec[name] for rec in records])

  def extract_metrics(self, rs, records, param_names):
    for name in param_names:
      Results.records_to_dict(rs, records, name)

  def extract_common_metrics(self, rs, records):
    self.extract_metrics(rs, records, self.common_metrics)
    if records:
      rs["pass_numbers"] = records[0]["pass_numbers"]

  def extract_margins(self, rs, records, margin_eps_index=-1):
    if records:
      if records[0]["soft_margins"].ndim == 3:
        rs["sq_margin"] = np.vstack(
            [rec["soft_margins"][:, :, margin_eps_index]**2 for rec in records])
      elif records[0]["soft_margins"].ndim == 2:
        rs["sq_margin"] = np.vstack(
            [rec["soft_margins"][:, margin_eps_index]**2 for rec in records])

  def get_metrics(self,
                  dataset,
                  groupby_param_name,
                  metrics,
                  bad_min,
                  extra_constraints=""):
    """Retrieves list of records, where each record is a dict.

    Args:
      dataset: name of a dataset.
      groupby_param_name: parameter to group results by.
      metrics: list of metric names to retrieve.
      bad_min: [bool] retrieve results for "bad" minumum experiment.
      extra_constraints: extra "WHERE" constraints.

    Returns:
      list of dicts, where each dict stands for a record.
    """

    metrics.extend(self.common_metrics)

    if self.mean_metrics:
      tf.logging.info("Retrieving means of metrics.")
      select_metrics = ["mean(%s, 0) as \"%s\"" % (s, s) for s in metrics]
      select_metrics.extend(
          ["stddev(%s, 0) as \"stddev_%s\"" % (s, s) for s in metrics])
    else:
      select_metrics = metrics

    args = dict(
        dataset=dataset,
        groupby_param_name=groupby_param_name,
        metrics=", ".join(select_metrics),
        metric_constraint=" and ".join(["%s is not null" % m for m in metrics]),
        bad_min=int(bad_min),
        extra_constraints=extra_constraints)

    db = SqliteDB(os.path.join(self.db_filename))
    if groupby_param_name in ["width", "depth"]:
      args["groupby_param_name"] = "network"

    sql_query = """
    SELECT pass_numbers,
    %(groupby_param_name)s,    

    %(metrics)s

    FROM rs
    WHERE dataset = "%(dataset)s"
          AND bad_min = %(bad_min)s
          AND %(metric_constraint)s
          %(extra_constraints)s
    GROUP by %(groupby_param_name)s
    ORDER by %(groupby_param_name)s
    """ % args

    tf.logging.info(sql_query)
    rs = db.execute(sql_query)

    # Handling width and depth parameter (for MLP) in a special way
    # i.e. parsing the name and convering into integer column
    if groupby_param_name == "width":
      for (i, rec) in enumerate(rs):
        layer_widths = rec["network"].split("_")[1].split("-")
        assert len(layer_widths) == 2
        assert layer_widths[0] == layer_widths[1]

        rs[i]["width"] = int(layer_widths[0])

      rs.sort(key=lambda x: x["width"])
    elif groupby_param_name == "depth":
      for (i, rec) in enumerate(rs):
        layer_widths = rec["network"].split("_")[1].split("-")
        rs[i]["depth"] = int(len(layer_widths))

      rs.sort(key=lambda x: x["depth"])

    return rs

  def get_sharpness(self, dataset, groupby_param_name, bad_min,
                    extra_constraints):
    """Get sharpness records.

    Retrieves records with common metrics (e.g. training/testing error)
    and sharpness metric.

    Args:
      dataset: name of a dataset.
      groupby_param_name: parameter to group results by.
      bad_min: [bool] retrieve results for "bad" minumum experiment.
      extra_constraints: extra "WHERE" constraints.

    Returns:
      list of dicts, where each dict stands for a record.
    """
    rs = dict()
    records = self.get_metrics(dataset, groupby_param_name,
                               ["sharpness", "alpha"], bad_min,
                               extra_constraints)

    for rec in records:
      alphas = rec["alpha"]
      if alphas.ndim == 2:
        alphas = alphas[0, :]

      if records[0]["sharpness"].ndim == 3:
        for i in range(len(alphas)):
          rs["sharpness_%s" % alphas[i]] = np.vstack(
              [rec["sharpness"][:, :, i].squeeze() for rec in records])
      elif records[0]["sharpness"].ndim == 2:
        for i in range(len(alphas)):
          rs["sharpness_%s" % alphas[i]] = np.vstack(
              [rec["sharpness"][:, i].squeeze() for rec in records])

    return rs

  def get_all_metrics(self,
                      dataset,
                      groupby_param_name,
                      bad_min=False,
                      extra_constraints=""):
    """Get records for all the metrics.

    Args:
      dataset: name of a dataset.
      groupby_param_name: parameter to group results by.
      bad_min: [bool] retrieve results for "bad" minumum experiment.
      extra_constraints: extra "WHERE" constraints.

    Returns:
      list of dicts, where each dict stands for a record.
    """

    # Pulling norm-metrics
    records = self.get_metrics(dataset, groupby_param_name, self.norm_metrics,
                               bad_min, extra_constraints)

    rs = dict()

    self.extract_common_metrics(rs, records)
    self.extract_metrics(rs, records, self.norm_metrics)
    self.extract_metrics(rs, records, [groupby_param_name])
    self.extract_margins(rs, records)

    # Pulling sharpness
    sharpness_rs = self.get_sharpness(dataset, groupby_param_name, bad_min,
                                      extra_constraints)
    rs.update(sharpness_rs)

    # Pulling Hessian spectral norm
    hessian_records = self.get_metrics(dataset, groupby_param_name,
                                       ["hessian_top_sv_means"], bad_min,
                                       extra_constraints)
    self.extract_metrics(rs, hessian_records, ["hessian_top_sv_means"])

    return rs

  def get_hessian(self,
                  dataset,
                  groupby_param_name,
                  bad_min=False,
                  extra_constraints=""):
    """Get Hessian spectral norm records.

    Retrieves records with common metrics (e.g. training/testing error)
    and the Hessian spectral norm metric.

    Args:
      dataset: name of a dataset.
      groupby_param_name: parameter to group results by.
      bad_min: [bool] retrieve results for "bad" minumum experiment.
      extra_constraints: extra "WHERE" constraints.

    Returns:
      list of dicts, where each dict stands for a record.
    """

    records = self.get_metrics(
        dataset, groupby_param_name,
        ["hessian_top_sv_means", "soft_margins", "train_grad_norm"], bad_min,
        extra_constraints)

    rs = dict()

    self.extract_common_metrics(rs, records)
    self.extract_metrics(rs, records, ["hessian_top_sv_means",
                                       "stddev_hessian_top_sv_means",
                                       "train_grad_norm"])
    self.extract_metrics(rs, records, [groupby_param_name])
    self.extract_margins(rs, records)

    return rs


class MetricTable(object):
  """Implements conversion of metric results to a LaTeX table."""

  def __init__(self,
               db_filename,
               dataset,
               groupby_param_name,
               network,
               extra_constraints=""):
    """Constructor.

    Args:
      db_filename: path to sqlite3 database.
      dataset: name of a dataset.
      groupby_param_name: parameter to group results by.
      network: network name.
      extra_constraints: extra "WHERE" constraints.
    """

    rs = Results(db_filename)

    extra_constraints_sql = ("and network like '%s' %s " % (network,
                                                            extra_constraints))

    self.records_good_min = rs.get_all_metrics(
        dataset,
        groupby_param_name,
        bad_min=False,
        extra_constraints=extra_constraints_sql)

    self.records_bad_min = rs.get_all_metrics(
        dataset,
        groupby_param_name,
        bad_min=True,
        extra_constraints=extra_constraints_sql)

    self.n_params = self.records_good_min["sq_margin"].shape[0]

  @staticmethod
  def format_number(num):
    """Formats a float.

    Args:
      num: float value.

    Returns:
      if num is in [1e-2, 1e+3), returns a float with 10^-2 precision.
      Otherwise returns value in scientific format.
    """
    if 1e-2 <= num < 1e+3:
      return "$%.2f$" % num
    elif num == 0.0:
      return "$0$"
    else:
      base, exponent = ("%.1e" % num).split("e")
      return "$%s \\cdot 10^{%s}$" % (base, int(exponent))

  @staticmethod
  def extract_column(records_good_min_metric, records_bad_min_metric):
    """Formats a column of a LaTeX table.

    Given a numpy array of records corresponding to good-minumum experiment,
    and a bad one, formats these into two adjacent columns.
    Highlights minimal and maximal value in the "bad" column.

    Args:
      records_good_min_metric: numpy array of values from a "good" experiment.
      records_bad_min_metric: numpy array of values from a "bad" experiment.

    Returns:
      a string in LaTeX format.
    """
    min_i_bad_min_metric = np.argmin(records_bad_min_metric)
    max_i_bad_min_metric = np.argmax(records_bad_min_metric)
    column = []
    for (i, z) in enumerate(
        zip(records_good_min_metric, records_bad_min_metric)):
      if min_i_bad_min_metric == i or max_i_bad_min_metric == i:
        zs = tuple(map(MetricTable.format_number, z))
        column.append("%s & \\boldmath{%s}" % zs)
      else:
        column.append("%s & %s" % tuple(map(MetricTable.format_number, z)))

    return column

  def print(self, metrics, normalize_by_margin=False, print_full_doc=False):
    """Formats a latex table for a given set of metrics.

    Args:
      metrics: list of metric names.
      normalize_by_margin: normalize metrics by the squared soft margin.
      print_full_doc: wrap LaTeX table into the markup ready for compilation.

    Returns:
      a table formatted as a LaTeX string.
    """
    pass_numbers = self.records_good_min["pass_numbers"]

    columns = []

    good_sq_soft_margin = self.records_good_min["sq_margin"].squeeze()
    bad_sq_soft_margin = self.records_bad_min["sq_margin"].squeeze()

    # Subselect index is passed whenever one record is a vector
    # e.g. eigenvalues of all layers
    for (metric_name, metric_label, subselect_index) in metrics:
      records_good_min_metric = self.records_good_min[metric_name].squeeze()
      records_bad_min_metric = self.records_bad_min[metric_name].squeeze()

      # Backwards compatibility
      # older experiments recorded multiple "bad" minima snapshots
      # here we are keeping only the last one
      if records_bad_min_metric.ndim == 2:
        records_bad_min_metric = records_bad_min_metric[-1, :]

      if subselect_index:
        records_good_min_metric = records_good_min_metric[:, subselect_index]
        records_bad_min_metric = records_bad_min_metric[:, subselect_index]

      if normalize_by_margin and (metric_name != "sq_margin"):
        records_good_min_metric /= good_sq_soft_margin
        records_bad_min_metric /= bad_sq_soft_margin

      column = MetricTable.extract_column(records_good_min_metric,
                                          records_bad_min_metric)
      columns.append(column)

    val_error_good = self.records_good_min["val_error"].squeeze()
    val_error_bad = self.records_bad_min["val_error"].squeeze()
    train_error_good = self.records_good_min["train_error"].squeeze()
    train_error_bad = self.records_bad_min["train_error"].squeeze()

    val_zero_one_error_good = (
        self.records_good_min["val_zero_one_error"].squeeze())
    val_zero_one_error_bad = (
        self.records_bad_min["val_zero_one_error"].squeeze())
    train_zero_one_error_good = (
        self.records_good_min["train_zero_one_error"].squeeze())
    train_zero_one_error_bad = (
        self.records_bad_min["train_zero_one_error"].squeeze())

    # Backwards compatibility again
    if val_error_bad.ndim == 2:
      val_error_bad = val_error_bad[-1, :]
      train_error_bad = train_error_bad[-1, :]
      val_zero_one_error_bad = val_zero_one_error_bad[-1, :]
      train_zero_one_error_bad = train_zero_one_error_bad[-1, :]

    error_metrics_all = [[(train_error_bad, train_error_good, "Train error"),
                          (val_error_bad, val_error_good, "Val error")],
                         [(train_zero_one_error_bad, train_zero_one_error_good,
                           "Train error (0/1)"),
                          (val_zero_one_error_bad, val_zero_one_error_good,
                           "Val error (0/1)")]]
    error_labels = []
    for (i, error_metrics) in enumerate(error_metrics_all):
      for (metric_bad, metric_good, label) in error_metrics:
        column = MetricTable.extract_column(metric_good, metric_bad)
        columns.append(column)
        error_labels.append(label)

    rows = zip(*columns)

    table_lines = []

    if print_full_doc:
      table_lines.append(r"\documentclass{article}")
      table_lines.append(
          r"\usepackage[a4paper, landscape, margin=2mm]{geometry}")
      table_lines.append(
          r"\usepackage{amsmath,amssymb,amsfonts,amsthm,graphics}")
      table_lines.append(r"\begin{document}")
      table_lines.append(r"\begin{center}")

    table_lines.append(r"\begin{table}")
    table_lines.append(r"\scalebox{0.6}{")
    table_lines.append(r"\begin{tabular}{%s|}" % ("|l" *
                                                  (2 * (len(metrics) + 4) + 1)))

    heads = (["Epoch"] + [
        r"\multicolumn{2}{|p{3cm}|}{%s}" % metric_label
        for (_, metric_label, _) in metrics
    ] + [
        r"\multicolumn{2}{|p{3cm}|}{%s}" % error_label
        for error_label in error_labels
    ])

    table_lines.append(r"\hline")
    table_lines.append(" & ".join(heads) + r" \\")
    table_lines.append(r"\hline")
    table_lines.append(" & ".join([""] + ["Good", "Bad"] *
                                  (len(metrics) + 4)) + r"\\ ")
    table_lines.append(r"\hline")
    table_lines.append("\n".join([
        " & ".join([str(pass_numbers[i])] + list(row)) + r" \\" + "\n\\hline"
        for (i, row) in enumerate(rows)
    ]))

    table_lines.append(r"\end{tabular}")
    table_lines.append(r"}")
    table_lines.append(r"\end{table}")

    if print_full_doc:
      table_lines.append(r"\end{center}")
      table_lines.append(r"\end{document}")

    return "\n".join(table_lines)


class HessianVsMarginPlot(object):
  """Plots experimental results with Hessian spectral norm and margin."""

  def __init__(self, db_filename, dataset, network, groupby_param_name,
               figure_path):
    rs = Results(db_filename, mean_metrics=True)
    """Constructor.

    Arranges results into a plot table, where columns are
    progressions of various metrics over epochs, and rows
    are different settings of a "groupby" parameter (e.g. a learning rate).


    db_filename: path to sqlite3 database.
    dataset: dataste name.
    network: network name.
    groupby_param_name: parameter to group results by.
    network: network name.
    figure_path: path for a PDF file with resulting figure.
    """

    self.records_good_min = rs.get_hessian(
        dataset,
        groupby_param_name,
        bad_min=False,
        extra_constraints="and network like '%s' " % network)

    self.records_bad_min = rs.get_hessian(
        dataset,
        groupby_param_name,
        bad_min=True,
        extra_constraints="and network like '%s' " % network)

    self.groupby_param_name = groupby_param_name
    self.n_params = self.records_good_min["sq_margin"].shape[0]
    self.figure_path = figure_path

  def plot_one_setting(self, param_index, legend=True):
    """Plot results for one "groupby" parameter.

    Args:
      param_index: index of a "groupby" parameter.
      legend: [bool] plot legend.
    """
    pass_numbers = self.records_good_min["pass_numbers"]

    hessian_sv_good = self.records_good_min["hessian_top_sv_means"][
        param_index, :]
    sq_margin_good = self.records_good_min["sq_margin"][param_index, :]
    val_error_good = self.records_good_min["val_error"][param_index, :]
    train_error_good = self.records_good_min["train_error"][param_index, :]
    val_zero_one_error_good = self.records_good_min["val_zero_one_error"][
        param_index, :]
    train_zero_one_error_good = self.records_good_min["train_zero_one_error"][
        param_index, :]
    train_grad_norm_good = self.records_good_min["train_grad_norm"][
        param_index, :]

    hessian_sv_bad = self.records_bad_min["hessian_top_sv_means"][
        param_index, :]
    sq_margin_bad = self.records_bad_min["sq_margin"][param_index, :]
    val_error_bad = self.records_bad_min["val_error"][param_index, :]
    train_error_bad = self.records_bad_min["train_error"][param_index, :]
    val_zero_one_error_bad = self.records_bad_min["val_zero_one_error"][
        param_index, :]
    train_zero_one_error_bad = self.records_bad_min["train_zero_one_error"][
        param_index, :]
    train_grad_norm_bad = self.records_bad_min["train_grad_norm"][
        param_index, :]

    self.n_cell_rows = 5

    title = "%s = %s" % (self.groupby_param_name,
                         self.records_good_min[
                             self.groupby_param_name][param_index][0])

    self.plot_cell(
        param_index + 1,
        pass_numbers,
        hessian_sv_bad,
        hessian_sv_good,
        "||Hessian (bad)||_2",
        "||Hessian (good)||_2",
        title,
        plotter=plt.loglog,
        add_legend=legend)

    self.plot_cell(
        param_index + 1,
        pass_numbers,
        train_grad_norm_bad,
        train_grad_norm_good,
        "||grad (bad)||",
        "||grad (good)||",
        title,
        plotter=plt.loglog,
        add_legend=legend)

    self.plot_cell(
        self.n_params + param_index + 1,
        pass_numbers,
        1.0 / sq_margin_bad,
        1.0 / sq_margin_good,
        "Inv. of margin^2 (bad)",
        "Inv. of margin^2 (good)",
        title,
        plotter=plt.loglog,
        add_legend=legend)

    self.plot_cell(
        2 * self.n_params + param_index + 1,
        pass_numbers,
        train_error_bad,
        train_error_good,
        "Train error (bad)",
        "Train error (good)",
        title,
        plotter=plt.semilogx,
        add_legend=legend)

    self.plot_cell(
        2 * self.n_params + param_index + 1,
        pass_numbers,
        val_error_bad,
        val_error_good,
        "Val error (bad)",
        "Val error (good)",
        title,
        plotter=plt.semilogx,
        add_legend=legend)

    self.plot_cell(
        3 * self.n_params + param_index + 1,
        pass_numbers,
        train_zero_one_error_bad,
        train_zero_one_error_good,
        "Train (0/1) error (bad)",
        "Train (0/1) error (good)",
        title,
        plotter=plt.semilogx,
        add_legend=legend)

    self.plot_cell(
        3 * self.n_params + param_index + 1,
        pass_numbers,
        val_zero_one_error_bad,
        val_zero_one_error_good,
        "Val 0/1 error (bad)",
        "Val 0/1 error (good)",
        title,
        plotter=plt.semilogx,
        add_legend=legend)

  def plot_cell(self,
                i,
                x,
                y_bad,
                y_good,
                label_bad,
                label_good,
                title,
                plotter=plt.plot,
                add_legend=True):
    """Plot one cell of a plot table.

    Args:
      i: subplot index of a cell.
      x: values on the x axis.
      y_bad: values on the y axis, for a "bad" experiment.
      y_good: values on the y axis, for a "good" experiment.
      label_bad: corresponding label.n
      label_good: corresponding label.
      title: title of a plot.
      plotter: matplotlib plotting function.
      add_legend: [bool] plot a legend.
    """
    if any(np.isnan(y_bad)) or any(np.isnan(y_good)):
      tf.logging.info("i=%d, Cannot plot: contains NaNs." % i)
      return

    ax = plt.subplot(self.n_cell_rows, self.n_params, i)
    ax.set_title(title)

    plot_rs = plotter(x, y_bad, linewidth=3, label=label_bad)
    plotter(
        x,
        y_good,
        linewidth=3,
        label=label_good,
        color=plot_rs[0].get_color(),
        linestyle="--")

    if add_legend:
      legend = plt.legend(loc="best", fontsize="small")
      legend = legend.get_frame().set_alpha(0.5)

    plt.grid(True)

  def plot(self):
    plt.figure(figsize=(self.n_params * 10, 10))

    for i in range(self.n_params):
      self.plot_one_setting(i, legend=(i == 0))

    tf.logging.info("Saving to %s", self.figure_path)
    plt.savefig(self.figure_path, tight_layout=True, bbox_inches="tight")


def pdflatex(tex, pdf_path):
  _, fname = tempfile.mkstemp()
  open(fname, "wt").write(tex)

  shell = ("pdflatex --jobname='%s' --output-directory='%s' %s" %
           (os.path.basename(pdf_path).split(".")[0], os.path.dirname(pdf_path),
            fname))
  pdflatex_out = os.popen(shell).read()
  tf.logging.info(pdflatex_out)


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.do == "plot_hessian_vs_margin":
    plot = HessianVsMarginPlot(FLAGS.db, FLAGS.dataset, FLAGS.network,
                               FLAGS.groupby_param, FLAGS.pdf)

    plot.plot()
    if FLAGS.show:
      plt.show()

  elif FLAGS.do.startswith("table_"):
    if FLAGS.do == "table_norms":
      metrics = [("weight_l2_norms", "Weight L2 norm", None),
                 ("path_l2_norms", "Path L2 norm", None),
                 ("spectral_products", "Prod. of layer\\newline spectral norms",
                  None),
                 ("spectral_complexities", "Spectral\\newline complexity",
                  None),
                 ("train_grad_norm", "Train grad. norm", None),
                 ("val_grad_norm", "Val grad. norm", None)]
    elif FLAGS.do == "table_phased":
      metrics = [("weight_variance", "Weight variance", None),
                 ("hessian_top_sv_means", "Hessian\\newline spectral norm",
                  None),
                 ("train_grad_norm", "Train grad. norm", None),
                 ("val_grad_norm", "Val grad. norm", None),
                 ("sharpness_0.0005",
                  "Sharpness\\newline (alpha=0.0005)", None),
                 ("weight_entropy", "Weight entropy\\newline($10^3$ bin hist)",
                  None), ("sq_margin", "Squared\\newline soft margin", None)]

    table = MetricTable(FLAGS.db, FLAGS.dataset, "bad_min", FLAGS.network,
                        "and learning_rate=0.05")
    table_text = table.print(metrics, print_full_doc=FLAGS.pdf)

    if FLAGS.pdf:
      pdflatex(table_text, FLAGS.pdf)
    else:
      print(table_text)


if __name__ == "__main__":
  tf.app.run(main)
