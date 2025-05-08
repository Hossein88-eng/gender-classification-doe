import tensorflow as tf

def load_datasets(data_dir, img_size=130, batch_size=32, validation_split=0.1, test_split=0.0, seed=101):
    """Load and preprocess datasets from a directory structure."""
    image_size = (img_size, img_size)
    total_split = validation_split + test_split

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary',
        validation_split=total_split,
        subset="training",
        seed=seed
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        directory=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode='binary',
        validation_split=total_split,
        subset="validation",
        seed=seed
    )

    val_batches = tf.data.experimental.cardinality(val_dataset)
    test_size = round(val_batches.numpy() * (test_split / total_split))
    test_dataset = val_dataset.take(test_size)
    val_dataset = val_dataset.skip(test_size)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    return train_dataset, val_dataset, test_dataset
    