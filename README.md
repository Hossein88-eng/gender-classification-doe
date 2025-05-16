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
Train the model inside a GitHub Codespace

Provided sample DOE design point's notebook (DOE_Run 06.ipynb) uses the following internal custom Python modules to simplify, reproducable, and scalable data loading, CNN model building, training and evaluation as well as the prediction for the classes of new images:
```bash
src.data_loader
src.model_builder
src.training_utils
src.image_predictor
```

Post-DOE model can be run via provided notebooks for the selected design points in the notebooks section.

## Results
Final model performance and design matrix is available in the csv files as well as the reported result files. Report file is also available in the results section (Results of DOE and Post-DOE Studies.pdf).

## License
See LICENSE file in the project repository as well as the README file in the data folder for details.

## Further Information & Contact
For further information and contact to the author, you can send an email via the following email address: beidaghydizaji@gmx.de 
