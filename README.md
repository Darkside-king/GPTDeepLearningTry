# Device Performance Prediction

This project demonstrates a simple deep learning workflow for predicting a device's overall performance based on benchmark scores.

## Contents
- `data/generate_data.py`: script to generate a synthetic dataset of benchmark scores.
- `train.py`: trains a neural network model using the data.
- `predict.py`: loads the model to predict performance for new devices and flags anomalous inputs.

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate sample data:
   ```bash
   python data/generate_data.py --samples 500 --output data/sample_performance.csv
   ```
3. Train the model:
   ```bash
   python train.py --data data/sample_performance.csv --epochs 20
   ```
4. Predict on new data:
   ```bash
   python predict.py path/to/your.csv --model model.pt --stats stats.pt
   ```
