import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def train_model(model, train_ds, val_ds, callbacks, epochs=15, verbose=2):
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2)
    return pd.DataFrame(history.history)


def plot_metrics(history_df, run=6):
    """ "Plotting history of the fitted model."""
    plt.figure(figsize=(6, 4))
    history_df[["loss", "val_loss"]].plot()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"history_loss of run {run}", dpi=300)
    plt.show()

    plt.figure(figsize=(6, 4))
    history_df[["accuracy", "val_accuracy"]].plot()
    plt.title("Accuracy over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"history_accuracy of run {run}", dpi=300)
    plt.show()


def evaluate_model(model, val_ds):
    """Evaluates the model and returns classification report
    & confusion matrix."""
    y_true = np.concatenate([y.numpy() for _, y in val_ds])
    y_pred_prob = model.predict(val_ds)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()

    report = classification_report(
        y_true, y_pred, target_names=[
            "Female", "Male"])
    conf_matrix = confusion_matrix(y_true, y_pred)

    return report, conf_matrix


def plot_confusion_matrix(conf_matrix):
    """Plots the confusion matrix as a heatmap."""
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=["Female", "Male"],
        yticklabels=["Female", "Male"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")
    plt.show()
