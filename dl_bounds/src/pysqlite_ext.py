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

"""Tools for handling results stored in sqlite database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import io
import numpy as np
import sqlite3


def py_type2sql_type(v):
  if v is None:
    return 'numeric'
  if v.__class__.__name__ == 'ndarray' and len(v) == 1:
    return 'numeric'

  return {
      'bool': 'numeric',
      'int': 'numeric',
      'float': 'numeric',
      'str': 'text',
      'ndarray': 'array',
      'list': 'array',
      'matrix': 'array'
  }[v.__class__.__name__]


class Statistic(object):

  def __init__(self):
    self.elements = []
    self.axis = None

  def step(self, value, axis):
    value = convert_array(value)
    self.elements.append(value)
    if not self.axis:
      self.axis = axis
    else:
      assert self.axis == axis


class MeanFunc(Statistic):

  def finalize(self):
    return adapt_array(np.mean(self.elements, axis=self.axis))


class StddevFunc(Statistic):

  def finalize(self):
    return adapt_array(np.std(self.elements, axis=self.axis))


def adapt_array(arr):
  if len(arr) == 1:
    return float(arr)

  arr = arr.squeeze()
  out = io.BytesIO()
  np.save(out, arr)
  out.seek(0)
  return buffer(out.read())


def convert_array(text):
  out = io.BytesIO(str(text))
  out.seek(0)
  return np.load(out)


def sqlite_numpy_handling():
  # Converts np.array to TEXT when inserting
  sqlite3.register_adapter(np.ndarray, adapt_array)
  # Converts TEXT to np.array when selecting
  sqlite3.register_converter('array', convert_array)

  sqlite3.register_adapter(list, adapt_array)


def open_sqlite_db(location):
  sqlite_numpy_handling()
  conn = sqlite3.connect(
      location, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
  conn.create_aggregate('mean', 2, MeanFunc)
  conn.create_aggregate('stddev', 2, StddevFunc)

  sqlite3.enable_callback_tracebacks(True)

  return conn


class SqliteDB(object):
  """Simple wrapper for Sqlite database.

  This class can handle numpy arrays as records in the database.
  """

  def __init__(self, db_path):
    self.db_path = db_path
    self.conn = open_sqlite_db(self.db_path)

  def execute(self, query):
    """Executes query and returns records.

    Args:
      query: SQL query.

    Returns:
      list of dicts, one per record.
    """
    c = self.conn.cursor()
    c.execute(query)
    desc = [x[0] for x in c.description]
    rs = c.fetchall()
    for i in range(len(rs)):
      rs[i] = list(rs[i])
      for j in range(len(rs[i])):
        if isinstance(rs[i][j], buffer):
          rs[i][j] = convert_array(rs[i][j])

      rs[i] = dict(zip(desc, rs[i]))

    return rs
