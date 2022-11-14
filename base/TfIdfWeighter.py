import math

import numpy as np


class TfIdfWeighter:
    def __init__(self, cleaned_plots: []):
        self.plots = cleaned_plots
        self.n = len(self.plots)
        l = max([len(pl) for pl in self.plots])
        self.tf_idf = np.zeros((self.n, l))
        self.vocab = set()
        self._set_vocab()

    def calc_tf(self):
        for i, plot in enumerate(self.plots):
            _v = {}
            for word in plot:
                _v[word] = _v.get(word, 0) + 1

            for j, word in enumerate(plot):
                self.tf_idf[i, j] = _v.get(word, 0)

    def _set_vocab(self):
        for plot in self.plots:
            for word in plot:
                self.vocab.add(word)

    def get_tf_idf(self):
        self.calc_tf()
        word_sets = [set(plot) for plot in self.plots]
        idf = {}
        for word in self.vocab:
            fn = lambda _w, _s: 1 if _w in _s else 0
            df = sum([fn(word, word_set) for word_set in word_sets])
            idf[word] = math.log(self.n / (1 + df))

        for i, plot in enumerate(self.plots):
            for j, word in enumerate(plot):
                self.tf_idf[i, j] *= (idf.get(word, 0))

        return self.tf_idf











