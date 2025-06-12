import argparse
import pandas as pd
import torch
from model import PerformancePredictor

MODEL_PATH = 'model.pt'
STATS_PATH = 'stats.pt'

def load_stats():
    data = torch.load(STATS_PATH, weights_only=False)
    return data['mean'], data['std']

def check_anomaly(features, mean, std, threshold=3.0):
    z = torch.abs((features - mean) / std)
    return torch.any(z > threshold, dim=1)

def predict(csv_path: str):
    df = pd.read_csv(csv_path)
    feats = df[['cpu_score', 'memory_score', 'disk_score', 'network_score']].values
    mean, std = load_stats()
    norm_feats = (feats - mean) / std
    x = torch.tensor(norm_feats, dtype=torch.float32)
    model = PerformancePredictor()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    with torch.no_grad():
        preds = model(x).numpy()
    anomalies = check_anomaly(torch.tensor(feats, dtype=torch.float32), mean, std)
    df['predicted_performance'] = preds
    df['anomaly'] = anomalies.numpy()
    print(df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict device performance')
    parser.add_argument('csv', help='CSV file with benchmark scores')
    args = parser.parse_args()
    predict(args.csv)
