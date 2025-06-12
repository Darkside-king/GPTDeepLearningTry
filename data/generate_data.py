import numpy as np
import pandas as pd

np.random.seed(0)

# Generate random benchmark scores
n_samples = 500
cpu = np.random.normal(loc=2500, scale=300, size=n_samples)
memory = np.random.normal(loc=1500, scale=200, size=n_samples)
disk = np.random.normal(loc=500, scale=50, size=n_samples)
network = np.random.normal(loc=1000, scale=150, size=n_samples)

# Overall performance is a weighted combination plus noise
performance = 0.4 * cpu + 0.3 * memory + 0.2 * disk + 0.1 * network
performance += np.random.normal(scale=100, size=n_samples)

# Create DataFrame
df = pd.DataFrame({
    'cpu_score': cpu,
    'memory_score': memory,
    'disk_score': disk,
    'network_score': network,
    'performance_score': performance
})

# Save to CSV
if __name__ == '__main__':
    df.to_csv('data/sample_performance.csv', index=False)
