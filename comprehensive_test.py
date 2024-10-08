import torch
import torch.nn as nn
import numpy as np
from torchqrnn import QRNN
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ---------- LORENZ ATTRACTOR GENERATION ----------
def lorenz_system(x, y, z, s=10, r=28, b=8/3):
    """
    Lorenz system differential equations.
    dx/dt = s * (y - x)
    dy/dt = x * (r - z) - y
    dz/dt = x * y - b * z
    """
    dx = s * (y - x)
    dy = x * (r - z) - y
    dz = x * y - b * z
    return dx, dy, dz


def generate_lorenz_data(T=10000, dt=0.01, init_state=(1.0, 1.0, 1.0)):
    """
    Generate Lorenz attractor data.
    T: Total time steps
    dt: Time step size
    init_state: Initial conditions (x0, y0, z0)
    """
    x, y, z = init_state
    trajectory = np.zeros((T, 3))
    
    for i in range(T):
        dx, dy, dz = lorenz_system(x, y, z)
        x += dx * dt
        y += dy * dt
        z += dz * dt
        trajectory[i] = [x, y, z]
    
    return trajectory


# ---------- DATASET AND DATALOADER ----------
class LorenzDataset(Dataset):
    def __init__(self, data, seq_len=50):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.seq_len = seq_len
    
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        return self.data[idx:idx + self.seq_len], self.data[idx + self.seq_len]


def get_dataloaders(data, seq_len=50, batch_size=64, split=0.8):
    """
    Create dataloaders for training and validation
    """
    split_idx = int(len(data) * split)
    train_data, val_data = data[:split_idx], data[split_idx:]

    train_dataset = LorenzDataset(train_data, seq_len=seq_len)
    val_dataset = LorenzDataset(val_data, seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------- MODEL DEFINITIONS ----------
class LSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


class QRNNModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=2):
        super(QRNNModel, self).__init__()
        self.qrnn = QRNN(input_size, hidden_size, num_layers=num_layers, dropout=0.2)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x):
        out, _ = self.qrnn(x)
        out = self.fc(out[:, -1, :])  # Take the output of the last time step
        return out


# ---------- TRAINING AND EVALUATION ----------
def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    model = model.to(device)
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
        
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses


# ---------- PLOTTING ----------
def plot_losses(train_losses, val_losses, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.savefig(filename)
    plt.close()


# ---------- MAIN EXECUTION ----------
if __name__ == '__main__':
    # Generate Lorenz attractor data
    print("Generating Lorenz attractor data...")
    data = generate_lorenz_data(T=10000, dt=0.01)

    # Prepare dataloaders
    train_loader, val_loader = get_dataloaders(data, seq_len=50, batch_size=64)

    # Instantiate models
    lstm_model = LSTMModel()
    qrnn_model = QRNNModel()

    # Train LSTM model
    print("Training LSTM model...")
    lstm_train_losses, lstm_val_losses = train_model(lstm_model, train_loader, val_loader, num_epochs=20)

    # Train QRNN model
    print("Training QRNN model...")
    qrnn_train_losses, qrnn_val_losses = train_model(qrnn_model, train_loader, val_loader, num_epochs=20)

    # Plot training losses
    plot_losses(lstm_train_losses, lstm_val_losses, "LSTM Training and Validation Losses", 'lstm_training')
    plot_losses(qrnn_train_losses, qrnn_val_losses, "QRNN Training and Validation Losses", 'qrnn_training')

    # Both models have now been trained on the Lorenz attractor data
