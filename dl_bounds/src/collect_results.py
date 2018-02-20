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

"""Implements collection of results from eval.py into sqlite database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import os
import pickle as pkl
import pprint
from dl_bounds.src.pysqlite_ext import open_sqlite_db
from dl_bounds.src.pysqlite_ext import py_type2sql_type
from dl_bounds.src.pysqlite_ext import SqliteDB
import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_enum("do", None, ["collect", "query"], "Command.")
tf.flags.DEFINE_string("source_dir", None, "Directory with pickled dicts.")
tf.flags.DEFINE_string("target_db", None, "Target DB file.")
tf.flags.DEFINE_string("sql_query", None, "SQL query for do=query.")

tf.flags.mark_flag_as_required("do")
tf.flags.mark_flag_as_required("source_dir")
tf.flags.mark_flag_as_required("target_db")


class DictsToSqlite(object):
  """Base class for collection."""

  def __init__(self, glob_expr, db_path):
    """Constructor.

    Args:
      glob_expr: glob expression that must match pickled dicts produced
        by eval.py
      db_path: target database path.
    """
    self.glob_expr = glob_expr
    self.db_path = db_path
    self.records = None

  def read_files(self):
    raise NotImplementedError

  def collect_files(self):
    if not self.records:
      self.records = self.read()
    else:
      return self.records

  def detect_columns(self):
    columns = set()

    for rec in self.records:
      columns.update([(k, py_type2sql_type(v)) for (k, v) in rec.items()])

    return columns

  def create_db(self):
    """Creates sqlite database."""

    self.collect_files()
    columns = self.detect_columns()

    conn = open_sqlite_db(self.db_path)
    c = conn.cursor()
    create_string = (
        "create table rs(id integer primary key autoincrement, %s)" %
        ",".join(["%s %s" % (k, v) for (k, v) in columns]))

    c.execute(create_string)

    for (i, rec) in enumerate(self.records):
      tf.logging.info("Processing file %d/%d..." % (i + 1, len(self.records)))
      keys, values = zip(*rec.items())
      values = list(values)

      # Converting lists to arrays
      for (i, v) in enumerate(values):
        if isinstance(v, list):
          values[i] = np.array(v)

      c.execute("insert into rs (%s) values (%s)" %
                (",".join(keys), ",".join(["?"] * len(keys))), values)

    conn.commit()


class LocalPickledDictsToSqlite(DictsToSqlite):

  def read(self):
    records = []

    for filename in glob.glob(self.glob_expr):
      records.append(pkl.load(open(filename, "rb")))

    return records


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if FLAGS.do == "collect":
    LocalPickledDictsToSqlite(
        os.path.join(FLAGS.source_dir, "*"), FLAGS.target_db).create_db()
  elif FLAGS.do == "query":
    db = SqliteDB(FLAGS.target_db)
    pprint.pprint(db.execute(FLAGS.sql_query))
  else:
    tf.logging.info("Unknown command (try 'collect' or 'query').")

if __name__ == "__main__":
  tf.app.run(main)
