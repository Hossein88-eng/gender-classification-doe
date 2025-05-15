import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.utils import load_img, img_to_array


def load_model(model_path):
    """Loads the trained model."""
    return tf.keras.models.load_model(model_path)


def predict_image(model, img_path, img_size):
    """Predicts gender based on the given image and returns confidence."""
    img = load_img(img_path, target_size=(img_size, img_size))
    final_img = img_to_array(img)
    final_img = np.expand_dims(final_img, axis=0)

    prediction = model.predict(final_img)
    result = "Female" if prediction > 0.5 else "Male"
    confidence = (
        prediction[0][0] * 100 if result == "Female" else (1 - prediction[0][0]) * 100
    )

    return result, confidence, final_img


def visualize_layers(model, final_img, img_size):
    """Visualizes CNN layers for the given image."""

    dummy_input = np.random.rand(1, img_size, img_size, 3)
    model.predict(dummy_input)

    visualization_model = Model(
        inputs=model.inputs, outputs=[layer.output for layer in model.layers]
    )
    successive_feature_maps = visualization_model.predict(final_img)
    layer_names = [layer.name for layer in model.layers]

    for layer_name, feature_map in zip(layer_names, successive_feature_maps):
        if len(feature_map.shape) == 4:
            n_features = feature_map.shape[-1]
            size = feature_map.shape[1]
            display_grid = np.zeros((size, size * n_features))

            for i in range(n_features):
                x = feature_map[0, :, :, i]
                x -= x.mean()
                x /= x.std() + 1e-8  # Normalize
                x *= 64
                x += 128
                x = np.clip(x, 0, 255).astype("uint8")  # Convert to image format
                display_grid[:, i * size : (i + 1) * size] = x

            scale = 20.0 / n_features
            plt.figure(figsize=(scale * n_features, scale))
            plt.title(layer_name)
            plt.grid(False)
            plt.imshow(display_grid, aspect="auto", cmap="cividis")
            plt.show()
