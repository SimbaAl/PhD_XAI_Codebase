# Layer Relevance Propagation-based XAI Scheme for Channel Estimation

GRACE (Gradient-based Relevance Analysis for Channel Estimation) is a Python package that implements Layer-wise Relevance Propagation (LRP) to analyze, explain, and filter neural network inputs based on their contribution to model performance in channel estimation tasks.

## Project Structure
```
PhD_Codebase/
├── src/                    # Source code
│   ├── models/            # Neural network model implementations
│   │   ├── __init__.py
│   │   └── dnn.py        # DNN model architecture
│   ├── lrp/               # LRP implementation
│   │   ├── __init__.py
│   │   └── core.py       # Core LRP functionality
│   ├── feature_analysis/  # Feature analysis and categorization
│   │   ├── __init__.py
│   │   ├── categorization.py
│   │   └── evaluation.py
│   └── utils/            # Utility functions
│       ├── __init__.py
│       ├── data.py      # Data loading utilities
│       ├── model.py     # Model management
│       └── paths.py     # Path management
├── experiments/          # Experiment scripts
│   ├── train_model.py
│   ├── compute_relevance.py
│   └── evaluate_features.py
├── config/              # Configuration files
├── data/               # Dataset storage
│   ├── raw/           # Raw dataset files
│   └── processed/     # Processed dataset files
├── results/           # Experimental results
│   ├── figures/       # Generated figures
│   └── metrics/       # Performance metrics
├── tests/             # Unit tests
├── setup.py          # Package installation setup
└── requirements.txt   # Package dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SimbaAl/PhD_XAI_Codebase.git
cd grace-channel-estimation
```

2. Create and activate a Conda environment (recommended):
```bash
conda create -n grace_env python=3.11
conda activate grace_env
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Dataset Requirements

The project is designed for vehicle-to-vehicle communication data with the following specifications:

- Channel Model: Vehicle-To-Vehicle Expressway Same Direction with Wall (VTV-SDWW)
- Vehicle Velocity: 100 km/h
- Doppler Frequency: 550 Hz
- Modulation: 16QAM
- SNR Range: 0-40 dB (5 dB increments)

### Data Organization
Place your data files in the following structure:
```
data/
├── raw/
│   ├── High_VTV_SDWW_16QAM_TRFI_DNN_training_dataset_40.mat
│   ├── High_VTV_SDWW_16QAM_TRFI_DNN_testing_dataset_0.mat
│   ├── High_VTV_SDWW_16QAM_TRFI_DNN_testing_dataset_5.mat
│   └── ...
└── processed/
```

## Usage

### 1. Training the Model

Train the DNN model using:
```bash
python experiments/train_model.py \
    --mobility High \
    --channel-model VTV_SDWW \
    --modulation 16QAM \
    --scheme TRFI \
    --training-snr 40 \
    --input-size 104 \
    --hidden-sizes 23 29 21 \
    --output-size 104 \
    --batch-size 128 \
    --epochs 100
```

### 2. Computing Relevance Scores

Compute LRP relevance scores using:
```bash
python experiments/compute_relevance.py \
    --mobility High \
    --channel-model VTV_SDWW \
    --modulation 16QAM \
    --scheme TRFI \
    --training-snr 40
```

### 3. Evaluating Features

Analyze feature importance and performance:
```bash
python experiments/evaluate_features.py \
    --mobility High \
    --channel-model VTV_SDWW \
    --modulation 16QAM \
    --scheme TRFI \
    --training-snr 40
```

## Key Components

### 1. DNN Model
- Architecture: 104 → 23 → 29 → 21 → 104
- Activation: ReLU
- Loss Function: MSE
- Optimizer: Adam

### 2. LRP Implementation
- Epsilon-LRP rule for relevance computation
- Conservation property maintenance
- Batch processing capability

### 3. Feature Categorization
Categories based on relevance scores:
- High Positive: > 0.6
- Low Positive: 0.2 - 0.6
- Neutral: 0.1 - 0.19
- Negative: < 0.1

### 4. Analysis Outputs
Results are organized in:
```
results/
├── {mobility}_{channel_model}_{modulation}_{scheme}/
│   ├── visualizations/
│   │   ├── performance/
│   │   │   ├── mse_trends.png
│   │   │   └── rmse_trends.png
│   │   └── contribution/
│   │       └── relative_contribution.png
│   ├── metrics/
│   │   ├── stability/
│   │   ├── performance/
│   │   └── relative/
│   └── reports/
│       └── summary_report.txt
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Requirements

- Python 3.11
- PyTorch
- NumPy
- Matplotlib
- Seaborn
- h5py
- scipy

## API Documentation

### src.models.DNN
Main model class implementing the channel estimation network.

### src.lrp.LRP
Implements Layer-wise Relevance Propagation algorithm.

### src.feature_analysis.FeatureCategorizer
Handles feature categorization based on relevance scores.

### src.feature_analysis.FeatureEvaluator
Evaluates model performance across different feature categories.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{ngorima2024neural,
  title={Neural Network-based Vehicular Channel Estimation Performance: Effect of Noise in the Training Set},
  author={Ngorima, S.A and Helberg, A.S.J and Davel, M.H},
  booktitle={Artificial Intelligence Research (SACAIR 2024)},
  series={Communications in Computer and Information Science},
  volume={1976},
  pages={172--186},
  year={2024}
}
```

## Contact

Simbarashe Aldrin Ngorima - aldringorima@gmail.com

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that is short and to the point. It lets people do anything they want with your code as long as they provide attribution back to you and don't hold you liable.

### Key Points of the MIT License:
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ℹ️ License and copyright notice required
- ❌ No Liability
- ❌ No Warranty

When using this work, please cite:
```bibtex
@inproceedings{ngorima2024neural,
    title={Neural Network-based Vehicular Channel Estimation Performance: Effect of Noise in the Training Set},
    author={Ngorima, S.A and Helberg, A.S.J and Davel, M.H},
    booktitle={Artificial Intelligence Research (SACAIR 2024)},
    series={Communications in Computer and Information Science},
    volume={1976},
    pages={172--186},
    year={2024}
}