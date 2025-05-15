import unittest
import numpy as np
import tensorflow as tf
from src.image_predictor import load_model, predict_image
from src.data_loader import load_datasets
from src.training_utils import evaluate_model

# Define constants for testing
IMG_SIZE = 130
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.00
SEED_NUM = 101

MODEL_PATH = "/workspaces/gender-classification-doe/" "gender_recognition_DOE_run6.h5"

TEST_IMAGE_PATH = "/workspaces/gender-classification-doe/tests/" "Hossein_10.jpg"


class TestProject(unittest.TestCase):

    def setUp(self):
        """Set up the model and datasets before running tests."""
        self.model = load_model(MODEL_PATH)
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(
            IMG_SIZE,
            BATCH_SIZE,
            VALIDATION_SPLIT,
            TEST_SPLIT,
            SEED_NUM,
            shuffle_buffer_size=1000,
        )

    def test_model_loading(self):
        """Ensure the model loads correctly."""
        self.assertIsInstance(self.model, tf.keras.Model)

    def test_image_prediction(self):
        """Check if the model predicts an image correctly."""
        result, confidence, _ = predict_image(self.model, TEST_IMAGE_PATH, IMG_SIZE)
        self.assertIn(result, ["Female", "Male"])
        self.assertGreaterEqual(confidence, 0)

    def test_dataset_loading(self):
        """Check if datasets load correctly."""
        self.assertTrue(self.train_dataset)
        self.assertTrue(self.val_dataset)
        self.assertTrue(self.test_dataset)

    def test_model_evaluation(self):
        """Ensure the evaluation function returns a valid confusion matrix."""
        report, conf_matrix = evaluate_model(self.model, self.val_dataset)
        self.assertIsInstance(conf_matrix, np.ndarray)


if __name__ == "__main__":
    unittest.main()
