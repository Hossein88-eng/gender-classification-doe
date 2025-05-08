import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


def train_model(model, train_ds, val_ds, callbacks, epochs=20):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    return pd.DataFrame(history.history)

def evaluate_model(model, val_ds):
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_pred_prob = model.predict(val_ds)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    report = classification_report(y_true, y_pred, target_names=['Female', 'Male'])
    return report

def plot_metrics(history_df):
    plt.figure(figsize=(6, 4))
    history_df[['loss', 'val_loss']].plot()
    plt.title('Loss over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(6, 4))
    history_df[['accuracy', 'val_accuracy']].plot()
    plt.title('Accuracy over epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
    