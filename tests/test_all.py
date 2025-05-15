import os
import unittest
from pathlib import Path

from src.image_predictor import load_model

MODEL_PATH = "gender_recognition_DOE_run6.h5"
DATA_PATH = Path.cwd() / "data"

def data_available():
    return DATA_PATH.exists()

class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model(MODEL_PATH)

        cls.data_exists = data_available()
        if cls.data_exists:
            from src.data_loader import load_datasets
            cls.train_dataset, cls.val_dataset, cls.test_dataset = load_datasets()
        else:
            cls.train_dataset = cls.val_dataset = cls.test_dataset = None

    def test_model_loading(self):
        """Test if model loads correctly."""
        self.assertIsNotNone(self.model)

    @unittest.skipUnless(data_available(), "Data directory not available, skipping dataset loading test.")
    def test_dataset_loading(self):
        """Test if datasets load correctly."""
        self.assertIsNotNone(self.train_dataset)
        self.assertIsNotNone(self.val_dataset)
        self.assertIsNotNone(self.test_dataset)

    @unittest.skipUnless(data_available(), "Data directory not available, skipping image prediction test.")
    def test_image_prediction(self):
        """Test if model predicts on an image correctly."""
        sample = next(iter(self.test_dataset.take(1)))[0]
        preds = self.model.predict(sample)
        self.assertEqual(preds.shape[0], sample.shape[0])

    @unittest.skipUnless(data_available(), "Data directory not available, skipping model evaluation test.")
    def test_model_evaluation(self):
        """Test if model evaluation returns valid output."""
        preds = self.model.predict(self.test_dataset)
        self.assertIsNotNone(preds)
