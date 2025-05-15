from pathlib import Path
import tensorflow as tf

# Dynamically set the correct data path
DATA_DIR = Path.cwd() / "data"

def get_image_counts():
    """Returns the number of images for each class."""
    data_dir_woman = os.path.join(DATA_DIR, "woman")
    data_dir_man = os.path.join(DATA_DIR, "man")
    return len(os.listdir(data_dir_woman)), len(os.listdir(data_dir_man))

def load_datasets(data_path=DATA_DIR):
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

    test_dataset = val_dataset  # Or load separately if needed

    return train_dataset, val_dataset, test_dataset
