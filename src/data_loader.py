from pathlib import Path
import tensorflow as tf

BASE_DIR = Path.cwd()
DATA_PATH = BASE_DIR / "data"

def load_datasets(data_path=DATA_PATH):
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        data_path,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(224, 224),
        batch_size=32
    )

    test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset

