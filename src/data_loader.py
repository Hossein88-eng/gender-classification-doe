import os
import tensorflow as tf

DATA_DIR = "/workspaces/gender-classification-doe/data"


def get_image_counts():
    """Returns the number of images for each class."""
    data_dir_woman = os.path.join(DATA_DIR, "woman")
    data_dir_man = os.path.join(DATA_DIR, "man")
    return len(os.listdir(data_dir_woman)), len(os.listdir(data_dir_man))


def load_datasets(
        img_size,
        batch_size,
        validation_split,
        test_split,
        seed_num,
        shuffle_buffer_size):
    """Loads training, validation, and test datasets."""
    image_size = (img_size, img_size)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        validation_split=validation_split + test_split,
        subset="training",
        seed=seed_num,
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="binary",
        validation_split=validation_split + test_split,
        subset="validation",
        seed=seed_num,
    )

    # Further split validation dataset to extract test dataset if needed
    val_batches = tf.data.experimental.cardinality(val_dataset)
    test_size = round(
        val_batches.numpy() * (test_split / (validation_split + test_split))
    )

    test_dataset = val_dataset.take(test_size)
    val_dataset = val_dataset.skip(test_size)

    return train_dataset, val_dataset, test_dataset


def optimize_datasets(
        train_dataset,
        val_dataset,
        test_dataset,
        shuffle_buffer_size):
    """Optimizes datasets for performance."""
    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = (
        train_dataset.cache()
        .shuffle(shuffle_buffer_size)
        .prefetch(buffer_size=AUTOTUNE)
    )
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset


def get_example_batch(training_dataset):
    """Retrieves the first batch of images and labels."""
    for images, labels in training_dataset.take(1):
        return images, labels
