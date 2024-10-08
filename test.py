import torch
from torchqrnn import QRNN

seq_len, batch_size, input_size, hidden_size = 10, 32, 100, 128
x = torch.randn(seq_len, batch_size, input_size).cuda()

qrnn = QRNN(input_size, hidden_size, num_layers=2, dropout=0.5).cuda()
output, hidden = qrnn(x)

print(output.size())  # Should be [seq_len, batch_size, hidden_size]
print(hidden.size())  # Should be [num_layers, batch_size, hidden_size]
