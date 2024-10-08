import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchqrnn import QRNN

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Benchmark configurations
input_size = 384
hidden_size = 384
seq_len = 400
batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
batch_sizes = list(range(1024, 2048, 64))
num_layers = 5
num_batches = 50  # Number of batches to run per benchmark

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- Model Definitions ----------
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=False)
    
    def forward(self, x):
        out, _ = self.gru(x)
        return out

class QRNNModel(nn.Module):
    def __init__(self):
        super(QRNNModel, self).__init__()
        self.qrnn = QRNN(input_size, hidden_size, num_layers=num_layers, dropout=0.2)
    
    def forward(self, x):
        out, _ = self.qrnn(x)
        return out

# ---------- Benchmark Function ----------
def benchmark_model(model, batch_size, num_batches=8):
    # Create random input data of shape (seq_len, batch_size, input_size)
    inputs = torch.randn(num_batches, seq_len, batch_size, input_size).to(device)

    # Use CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # Move the model to the appropriate device (GPU)
    model.to(device)
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        # Warm-up pass (helps stabilize timings)
        with torch.no_grad():
            model(inputs[-1])

        torch.cuda.synchronize()  # Ensure all kernels have finished

        # Start timing
        start_event.record()
        
        # Run the model num_batches times
        with torch.no_grad():
            for i in range(num_batches):
                model(inputs[i])

        # Stop timing
        end_event.record()
        
        # Wait for all kernels to finish and calculate the elapsed time
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)  # in milliseconds
    
    # Calculate throughput: total batches per second
    throughput = ((num_batches * batch_size) / (elapsed_time / 1000.0))  # convert ms to seconds
    
    return throughput

# ---------- Main Benchmark Loop ----------
def run_benchmark():
    lstm_model = LSTMModel()
    gru_model = GRUModel()
    qrnn_model = QRNNModel()

    batch_throughput_lstm = []
    batch_throughput_gru = []
    batch_throughput_qrnn = []

    for batch_size in batch_sizes:
        print(f"Benchmarking with batch size: {batch_size}")

        # LSTM benchmark
        throughput_lstm = benchmark_model(lstm_model, batch_size)
        batch_throughput_lstm.append(throughput_lstm)
        print(f"LSTM: {throughput_lstm:.2f} batches/sec")

        # GRU benchmark
        throughput_gru = benchmark_model(gru_model, batch_size)
        batch_throughput_gru.append(throughput_gru)
        print(f"GRU: {throughput_gru:.2f} batches/sec")

        # QRNN benchmark
        throughput_qrnn = benchmark_model(qrnn_model, batch_size)
        batch_throughput_qrnn.append(throughput_qrnn)
        print(f"QRNN: {throughput_qrnn:.2f} batches/sec")

    return batch_throughput_lstm, batch_throughput_gru, batch_throughput_qrnn

# ---------- Plot Results ----------
def plot_results(batch_sizes, throughput_lstm, throughput_gru, throughput_qrnn):
    plt.figure(figsize=(12, 8))
    plt.plot(batch_sizes, throughput_lstm, label='LSTM Throughput', marker='o')
    plt.plot(batch_sizes, throughput_gru, label='GRU Throughput', marker='o')
    plt.plot(batch_sizes, throughput_qrnn, label='QRNN Throughput', marker='o')

    plt.title('Throughput Comparison (Batches Processed per Second)')
    plt.xlabel('Batch Size')
    plt.ylabel('Batches per Second')
    plt.xscale('log')  # Logarithmic scale for batch sizes
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_results.png")

if __name__ == "__main__":
    print("Starting benchmark...")
    throughput_lstm, throughput_gru, throughput_qrnn = run_benchmark()
    print("Benchmark completed. Plotting results...")
    
    # Plot the benchmark results
    plot_results(batch_sizes, throughput_lstm, throughput_gru, throughput_qrnn)
