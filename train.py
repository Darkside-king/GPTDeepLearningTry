"""Train the performance prediction model."""

import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import PerformancePredictor

DATA_PATH = "data/sample_performance.csv"
MODEL_PATH = "model.pt"
STATS_PATH = "stats.pt"

def load_data(path: str = DATA_PATH):
    df = pd.read_csv(path)
    features = df[["cpu_score", "memory_score", "disk_score", "network_score"]].values
    target = df["performance_score"].values
    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    normalized = (features - mean) / std
    return torch.tensor(normalized, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), mean, std

def train(data_path: str, model_path: str, stats_path: str, epochs: int = 20):
    X, y, mean, std = load_data(data_path)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PerformancePredictor()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: loss={loss.item():.4f}')

    torch.save(model.state_dict(), model_path)
    torch.save({'mean': mean, 'std': std}, stats_path)
    print('Model saved to', model_path)

def main():
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--data', default=DATA_PATH, help='Training CSV file')
    parser.add_argument('--model', default=MODEL_PATH, help='Path to save the model')
    parser.add_argument('--stats', default=STATS_PATH, help='Path to save feature stats')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()

    train(args.data, args.model, args.stats, args.epochs)


if __name__ == '__main__':
    main()
