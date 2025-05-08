import unittest
from src.data_loader import load_datasets
from src.model_builder import create_and_compile_model
from tensorflow.keras.models import Model


class TestPipeline(unittest.TestCase):
    
    def test_data_loading(self):
        train_ds, val_ds, test_ds = load_datasets()
        self.assertGreater(len(list(train_ds)), 0)
        self.assertGreater(len(list(val_ds)), 0)

    def test_model_structure(self):
        model = create_and_compile_model()
        self.assertIsInstance(model, Model)
        self.assertEqual(model.output_shape[-1], 1)

if __name__ == '__main__':
    unittest.main()