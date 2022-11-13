from torchsummary import summary
from torch import nn


class ModelSpecParser:
    def __init__(self, spec, dim):
        self.dim = dim
        self.seq = None
        self.parse(spec)
        summary(self.seq, (1, dim))

    def parse(self, spec):
        self.seq = nn.Sequential()
        _seq = []

        for layer in spec:
            parts = layer.split('_')

            if parts[0] == 'l':
                # Linear layer
                assert len(parts) == 3
                _seq.append(nn.Linear(int(parts[1]), int(parts[2])))
            elif parts[0] == 'd':
                # Dropout layer
                assert len(parts) == 2
                _seq.append(nn.Dropout(p=float(parts[1])))
            elif parts[0] == 's':
                # Softmax layer
                assert len(parts) == 2
                _seq.append(nn.Softmax(dim=int(parts[1])))
            elif parts[0] == 'r':
                # ReLU layer
                assert len(parts) == 1
                _seq.append(nn.ReLU())
            elif parts[0] == 'bn':
                assert len(parts) == 2
                _seq.append(nn.B)

        self.seq = nn.Sequential(*_seq)


if __name__ == "__main__":
    parser = ModelSpecParser(spec=["l_50_125", "d_0.3", "r", "l_125_125", "d_0.3",  "r", "l_125_150", "d_0.3", "r", "l_150_110", "d_0.3", "r", "l_110_75", "l_75_10", "s_0"], dim=50)