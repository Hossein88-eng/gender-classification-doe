import os
import unittest
from pathlib import Path

DATA_DIR = Path.cwd() / "data"

class TestProject(unittest.TestCase):
    def setUp(self):
        if not DATA_DIR.exists():
            self.skipTest(f"Data directory '{DATA_DIR}' not found. Skipping dataset-dependent tests.")
        
        from src.data_loader import load_datasets
        from src.image_predictor import load_model, MODEL_PATH

        self.train_dataset, self.val_dataset, self.test_dataset = load_datasets(DATA_DIR)
        self.model = load_model(MODEL_PATH)


if __name__ == "__main__":
    unittest.main()
