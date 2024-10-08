# torchqrnn/qrnn.py
import torch
from torch import nn

from .forget_mult import ForgetMult

class QRNNLayer(nn.Module):
    def __init__(self, input_size, hidden_size=None, save_prev_x=False, zoneout=0, window=1, output_gate=True):
        super(QRNNLayer, self).__init__()

        assert window in [1, 2], "Only window sizes of 1 or 2 are supported"
        self.window = window
        self.input_size = input_size
        self.hidden_size = hidden_size if hidden_size else input_size
        self.zoneout = zoneout
        self.save_prev_x = save_prev_x
        self.prevX = None
        self.output_gate = output_gate

        self.linear = nn.Linear(self.window * self.input_size, 3 * self.hidden_size if self.output_gate else 2 * self.hidden_size)

    def reset(self):
        self.prevX = None

    def forward(self, X, hidden=None):
        seq_len, batch_size, _ = X.size()

        if self.window == 1:
            source = X
        elif self.window == 2:
            Xm1 = [self.prevX if self.prevX is not None else torch.zeros(1, batch_size, self.input_size, device=X.device, dtype=X.dtype)]
            if seq_len > 1:
                Xm1.append(X[:-1, :, :])
            Xm1 = torch.cat(Xm1, 0)
            source = torch.cat([X, Xm1], 2)

        Y = self.linear(source)
        if self.output_gate:
            Z, F, O = torch.chunk(Y, 3, dim=2)
        else:
            Z, F = torch.chunk(Y, 2, dim=2)

        Z = torch.tanh(Z)
        F = torch.sigmoid(F)

        if self.zoneout and self.training:
            mask = F.new_full(F.size(), 1 - self.zoneout)
            F = F * mask

        C = ForgetMult()(F, Z, hidden)
        if self.output_gate:
            O = torch.sigmoid(O)
            H = O * C
        else:
            H = C

        if self.window > 1 and self.save_prev_x:
            self.prevX = X[-1:, :, :].detach()

        return H, C[-1, :, :].unsqueeze(0)

class QRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0):
        super(QRNN, self).__init__()

        self.layers = nn.ModuleList([QRNNLayer(input_size if l == 0 else hidden_size, hidden_size) for l in range(num_layers)])
        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, input, hidden=None):
        next_hidden = []
        output = input

        for i, layer in enumerate(self.layers):
            output, h = layer(output, None if hidden is None else hidden[i])
            next_hidden.append(h)

            if self.dropout != 0 and i < len(self.layers) - 1:
                output = torch.nn.functional.dropout(output, p=self.dropout, training=self.training)

        next_hidden = torch.cat(next_hidden, 0)
        return output, next_hidden
