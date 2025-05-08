# gender-classification-doe
It is a professional gender classification project combining manual tuning, DOE analysis, and automated callback optimization.



# Gender Classification using CNN and DOE

This project explores gender classification from facial images using Convolutional Neural Networks (CNNs), with a Design of Experiments (DOE) methodology to optimize model architecture and hyperparameters.

## Directory Structure
```
gender-classification-doe/
├── data/
│   └── README.md
├── notebooks/
│   └── gender_classification_DOE.ipynb
├── src/
│   ├── data_loader.py
│   ├── model_builder.py
│   ├── training_utils.py
├── results/
│   └── DOE_CNN_Design.xlsx
├── README.md
├── requirements.txt
├── LICENSE
├── Makefile
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
Final model performance and design matrix is available in `results/DOE_CNN_Design.xlsx`

## License
See LICENSE file for details.