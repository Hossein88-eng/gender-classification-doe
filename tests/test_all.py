import os
import unittest
from pathlib import Path

from src.image_predictor import load_model
from src.data_loader import load_datasets

MODEL_PATH = "gender_recognition_DOE_run6.h5"
DATA_PATH = Path.cwd() / "data"

def data_available():
    return DATA_PATH.exists()

class TestProject(unittest.TestCase):
    def setUp(self):
        self.model = load_model(MODEL_PATH)
        
        if not data_available():
            self.skip_all_tests = True
        else:
            self.skip_all_tests = False
            self.train_dataset, self.val_dataset, self.test_dataset = load_datasets()

    def test_dummy(self):
        if self.skip_all_tests:
            self.skipTest("Skipping test because data directory not found.")
        
        # Otherwise perform a real test
        self.assertIsNotNone(self.model)
