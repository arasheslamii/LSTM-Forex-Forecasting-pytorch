
# LSTM Model for USD/JPY Forecasting and ONNX Export

This project leverages Long Short-Term Memory (LSTM) neural networks to predict USD/JPY currency pair prices. 
The model processes historical forex data (OHLC: Open, High, Low, Close) to predict future trends and aims to assist in algorithmic trading. 
Additionally, the trained model is exported in ONNX format for interoperability with other platforms.

## Table of Contents

- [Overview](#overview)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [ONNX Export](#onnx-export)
- [Results](#results)
- [Dependencies](#dependencies)
- [License](#license)
- [Contributing](#contributing)

## Overview

The goal of this project is to build a multi-layer LSTM model that predicts future prices based on historical data. 
The dataset consists of 60 days of minute-level OHLC data, and the LSTM model is trained to forecast the next 60 time steps. 
This project also supports ONNX export, making it easy to deploy the model on different platforms.

## Project Workflow

1. **Data Collection:**  
   Data is sourced from MetaTrader 5 and Yahoo Finance or provided as a CSV.

2. **Data Preprocessing:**  
   Data scaling using MinMaxScaler and reshaping for time-series inputs.

3. **Model Architecture:**  
   LSTM model with two hidden layers, ReLU activations, and dropout for regularization.

4. **Training & Evaluation:**  
   Model is trained over 50 epochs with batch-wise mean squared error (MSE) loss.

5. **Export & Deployment:**  
   Trained model is saved in both ONNX and PyTorch formats for further use.

## Installation

Make sure you have the required packages installed:

```bash
pip install torch onnx yfinance pandas scikit-learn matplotlib
```

## Usage

1. **Load the data:**  
   Modify the data path in the code to your local dataset if needed.

2. **Train the model:**  
   Run the training loop provided in the notebook.

3. **Evaluate the model:**  
   Use the test loop to see the performance metrics like MSE, MAE, and R² score.

4. **Export the model:**  
   Save the trained model in ONNX and PyTorch formats.

## Model Training and Evaluation

- **Training Loop:**  
  The model is trained using an Adam optimizer with a learning rate of 1e-5 and MSE loss.

- **Evaluation Metrics:**  
  - Mean Squared Error (MSE)  
  - Mean Absolute Error (MAE)  
  - R² Score

## ONNX Export

The model is exported to ONNX format for easier deployment:

```python
import torch.onnx as onnx
dummy_input = torch.randn(1, 144, 4).to(device)
onnx.export(model, dummy_input, "model.onnx", export_params=True, opset_version=12)
```

## Results

Below is an example plot showing the comparison between actual and predicted values:

- **Mean Squared Error:** [Value]  
- **Mean Absolute Error:** [Value]  
- **R² Score:** [Value]

## Dependencies

- Python 3.x  
- PyTorch  
- ONNX  
- scikit-learn  
- matplotlib  
- pandas  
- yfinance  

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have improvements or suggestions.
