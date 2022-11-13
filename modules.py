from parser import ModelSpecParser
from torch import nn


class BasicDeepLearner(nn.Module):
    def __init__(self, model_spec, alpha, input_dim):
        super().__init__()
        self.seq = ModelSpecParser(model_spec, input_dim).seq
        self.alpha = alpha
