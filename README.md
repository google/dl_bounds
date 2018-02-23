## Deep Learning Complexity Measure Experiments

This is the code that compares various complexity measures for deep neural
networks, recently proposed in the literature.

Evaluation relies on three tools:
* src/eval.py -- runs comparison experiments and stores them as pickled dicts.
* src/collect_results.py -- collects results produced by eval.py into a
  sqlite database.
* src/results.py -- plotting and table formatting tools that read from
  sqlite database.

bin/mnist_demo.sh showcases an examples of evaluation workflow.

