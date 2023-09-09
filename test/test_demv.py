import numpy as np
import pandas as pd
from demv import DEMV
import unittest


class TestDEMV(unittest.TestCase):
    def setUp(self):
        # Define some sample data for testing
        np.random.seed(42)
        num_samples = 100
        self.x = pd.DataFrame(
            {
                "feature1": np.random.rand(num_samples),
                "feature2": np.random.rand(num_samples),
                "sensitive_var1": np.random.randint(0, 2, num_samples),
                "sensitive_var2": np.random.randint(0, 2, num_samples),
            }
        )
        self.y = np.random.randint(0, 2, num_samples)

    def test_fit_transform(self):
        # Create an instance of the DEMV class
        demv = DEMV(sensitive_vars=["sensitive_var1", "sensitive_var2"])

        # Test the fit_transform method
        x_balanced, y_balanced = demv.fit_transform(self.x, self.y)

        # Check if the output has the correct shape
        self.assertEqual(x_balanced.shape[0], len(y_balanced))

        # Check if the output has the correct type
        self.assertIsInstance(x_balanced, pd.DataFrame)

        # Check if the output has the correct type
        self.assertIsInstance(y_balanced, np.ndarray)

    def test_get_iters(self):
        # Create an instance of the DEMV class
        demv = DEMV(sensitive_vars=["sensitive_var1", "sensitive_var2"])

        # Test the get_iters method before fit_transform
        self.assertEqual(demv.get_iters(), 0)

        # Test the get_iters method after fit_transform
        _, _ = demv.fit_transform(self.x, self.y)
        self.assertNotEqual(demv.get_iters(), 0)

    def test_get_disparities(self):
        # Create an instance of the DEMV class
        demv = DEMV(sensitive_vars=["sensitive_var1", "sensitive_var2"])

        # Test that the get_disparities method returns an empty array before fit transform
        disparities = demv.get_disparities()
        self.assertEqual(disparities, [])

        # Test that the get_disparities method returns a non-empty array after fit transform
        demv.fit_transform(self.x, self.y)
        disparities = demv.get_disparities()
        self.assertNotEqual(disparities, [])


if __name__ == "__main__":
    unittest.main()
