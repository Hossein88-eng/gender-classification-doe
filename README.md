[![Build and Test](https://github.com/Hossein88-eng/gender-classification-doe/actions/workflows/main.yml/badge.svg)](https://github.com/Hossein88-eng/gender-classification-doe/actions/workflows/main.yml)

# gender-classification-doe
It is a professional gender classification project combining manual tuning, DOE analysis, and automated callback optimization.



# Gender Classification using CNN and DOE

This project explores gender classification from facial images using Convolutional Neural Networks (CNNs), with a Design of Experiments (DOE) methodology as well as a Post-DOE optimization process using learning rate decy callback as well as validation accuracy early stopping control to optimize model architecture and hyperparameters.

## Directory Structure
```
gender-classification-doe/
├── CSV files/
│   └── DOE results.csv
│   └── Post-DOE results.csv
├── data/
│   └── README.md
├── notebooks/
│   └── Exploratory Data Analaysis (EDA).ipynb
│   └── gender_classification_DOE.ipynb
│   └── gender_classification_Post_DOE.ipynb
│   └── Figure 21.ipynb
├── src/
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── training_utils.py
├── results/
│   └── DOE_CNN_Design.xlsx
│   └── Project Information & Introduction.pdf
│   └── Results of DOE and Post-DOE Studies.pdf
├── build.yml
├── Dockerfile
├── LICENSE
├── Makefile
├── README.md
├── requirements.txt
```

## Requirements
Install all required packages using:
```bash
pip install -r requirements.txt
```

## Methodology
- **Data Source**: Pixabay images (see `data/README.md`)
- **CNN Model**: Custom-designed using TensorFlow/Keras
- **DOE Approach**: Grid experiments tuning filter sizes, dropout rates, dense units, etc.

## How to Run
Train the model inside a GitHub Codespace:
```bash
make train
```
Run evaluation and plot results:
```bash
make evaluate
make plots
```

## Results
Final model performance and design matrix is available in the csv files as well as the reported result files.

## License
See LICENSE file for details.
