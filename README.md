# Heartbeat Classifier

A deep learning project that uses a **1D Convolutional Neural Network (CNN)** to classify ECG heartbeats into different categories using the MIT-BIH Arrhythmia Database.

The project explores automated heartbeat classification, with particular attention to identifying abnormal heartbeat patterns and addressing class imbalance.

## Features

- ECG signal processing using the WFDB library.
- Automatic heartbeat segmentation and preprocessing.
- Four-class heartbeat classification: Normal, Supraventricular, Ventricular and Fusion.
- 1D CNN implemented using PyTorch.
- Comparison of baseline and class-weighted training.
- Evaluation using precision, recall, F1-score and confusion matrices.
- Interactive browser-based ECG signal visualiser.

## Tech Stack

- Python
- PyTorch
- NumPy
- WFDB
- HTML, CSS and JavaScript

## Dataset

This project uses the [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/1.0.0/), available through PhysioNet.

The dataset contains annotated ECG recordings used for extracting and classifying individual heartbeats.

## Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/xt67/Heartbeat-Classifier.git
cd Heartbeat-Classifier
```

**2. Install dependencies**

```bash
pip install -r requirements.txt
```

**3. Download and preprocess the dataset**

```bash
python -m heartbeat_classifier.data --download
```

**4. Train the models**

```bash
python -m heartbeat_classifier.train --epochs 20
```

**5. Run tests**

```bash
python -m unittest discover -s tests -v
```

## Project Structure

```text
Heartbeat-Classifier/
├── heartbeat_classifier/  # Data processing and model training
├── tests/                 # Unit tests
├── web/                   # Interactive ECG visualiser
├── RESULTS.md             # Implementation and experiment status
├── requirements.txt
└── README.md
```

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and must not be used for clinical decisions.

---

Developed by [xt67](https://github.com/xt67).
