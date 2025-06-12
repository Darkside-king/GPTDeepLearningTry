"""Utility to generate a synthetic benchmark dataset."""

import argparse
import numpy as np
import pandas as pd


def create_dataset(n_samples: int, seed: int = 0) -> pd.DataFrame:
    """Return a DataFrame with random benchmark scores."""
    rng = np.random.default_rng(seed)

    cpu = rng.normal(loc=2500, scale=300, size=n_samples)
    memory = rng.normal(loc=1500, scale=200, size=n_samples)
    disk = rng.normal(loc=500, scale=50, size=n_samples)
    network = rng.normal(loc=1000, scale=150, size=n_samples)

    performance = 0.4 * cpu + 0.3 * memory + 0.2 * disk + 0.1 * network
    performance += rng.normal(scale=100, size=n_samples)

    return pd.DataFrame(
        {
            "cpu_score": cpu,
            "memory_score": memory,
            "disk_score": disk,
            "network_score": network,
            "performance_score": performance,
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic benchmark data")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to generate")
    parser.add_argument(
        "--output", default="data/sample_performance.csv", help="Where to save the CSV"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    df = create_dataset(args.samples, args.seed)
    df.to_csv(args.output, index=False)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
