import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from model import PerformancePredictor

DATA_PATH = 'data/sample_performance.csv'
MODEL_PATH = 'model.pt'
STATS_PATH = 'stats.pt'

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    features = df[['cpu_score', 'memory_score', 'disk_score', 'network_score']].values
    target = df['performance_score'].values
    # Normalize features
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    normalized = (features - mean) / std
    return torch.tensor(normalized, dtype=torch.float32), torch.tensor(target, dtype=torch.float32), mean, std

def train():
    X, y, mean, std = load_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = PerformancePredictor()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}: loss={loss.item():.4f}')

    torch.save(model.state_dict(), MODEL_PATH)
    torch.save({'mean': mean, 'std': std}, STATS_PATH)
    print('Model saved to', MODEL_PATH)

if __name__ == '__main__':
    train()
