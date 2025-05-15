import os
import unittest
from pathlib import Path

from src.image_predictor import load_model

MODEL_PATH = "gender_recognition_DOE_run6.h5"
DATA_PATH = Path.cwd() / "data"

def data_available():
    return DATA_PATH.exists()

@unittest.skipUnless(data_available(), "Data directory not available, skipping dataset tests.")
class TestProject(unittest.TestCase):
    def setUp(self):
        self.model = load_model(MODEL_PATH)
        # Load datasets only if available
        from src.data_loader import load_datasets
        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets()

    def test_dataset_loading(self):
        self.assertIsNotNone(self.train_dataset)
        self.assertIsNotNone(self.val_dataset)
        self.assertIsNotNone(self.test_dataset)

    def test_image_prediction(self):
        sample = next(iter(self.test_dataset.take(1)))[0]
        preds = self.model.predict(sample)
        self.assertEqual(preds.shape[0], sample.shape[0])

    def test_model_loading(self):
        self.assertIsNotNone(self.model)

    def test_model_evaluation(self):
        preds = self.model.predict(self.test_dataset)
        self.assertIsNotNone(preds)
