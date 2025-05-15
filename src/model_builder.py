import tensorflow as tf


def build_augmentation_model(fill_mode="nearest"):
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.1, fill_mode=fill_mode),
        ]
    )


def create_and_compile_model(
    img_size=130,
    dropout=0.2,
    n_units_last_layer=4096,
    n_filters_l1=32,
    n_filters_l2=64,
    learning_rate=0.0005,
    loss="binary_crossentropy",
    metrics=["accuracy"],
):
    augmentation_layers = build_augmentation_model()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(shape=(img_size, img_size, 3)),
            augmentation_layers,
            tf.keras.layers.Rescaling(1.0 / 255),
            #    CONV_LAYER_1:
            tf.keras.layers.Conv2D(n_filters_l1, (4, 4), activation="linear"),
            tf.keras.layers.MaxPooling2D(2, 2),
            #    CONV_LAYER_2:
            tf.keras.layers.Conv2D(n_filters_l2, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            #    CONV_LAYER_3:
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            #    CONV_LAYER_4:
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(dropout),
            #    BEFORE_LAST_LAYER:
            tf.keras.layers.Dense(n_units_last_layer, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


def reset_model_weights(model):
    for layer in model.layers:
        if hasattr(layer, "kernel_initializer"):
            layer.kernel.assign(layer.kernel_initializer(layer.kernel.shape))
        if hasattr(layer, "bias_initializer"):
            layer.bias.assign(layer.bias_initializer(layer.bias.shape))


def get_callbacks(desired_train_accuracy=0.89, desired_val_accuracy=0.89):
    class EarlyStoppingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            train_accuracy = logs.get("accuracy")
            val_accuracy = logs.get("val_accuracy")
            if (
                train_accuracy >= desired_train_accuracy
                and val_accuracy >= desired_val_accuracy
            ):
                self.model.stop_training = True
                print("Reached desired accuracy so cancelling training!")

    def scheduler(epoch, lr):
        if epoch < 10:
            return lr
        elif epoch < 15:
            return lr / 2
        elif epoch < 30:
            return lr
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
    return [lr_callback, EarlyStoppingCallback()]
