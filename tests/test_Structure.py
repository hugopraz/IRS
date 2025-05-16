import os
import json
import unittest
import numpy as np
from irs.ir_structure import (
    gaussian,
    reconstruct_spectrum,
)

# Test data constants
SAMPLE_PEAKS = [(1000, 0.8, 50), (1500, 0.5, 30)]
SAMPLE_SPECTRA = {
    "Isocyanide": [(2100, 0.9, 40), (1200, 0.3, 20)],
    "Hydroxyl": [(3400, 0.7, 60)]
}

class TestIRStructureFunctions(unittest.TestCase):
    # --- Gaussian Function Tests ---
    
    # Tests basic properties of the Gaussian function including symmetry and peak value
    def test_gaussian(self):
        x = np.array([0, 1, 2])
        result = gaussian(x, center=1, intensity=1.0, sigma=0.5)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertEqual(result[0], result[2])
        self.assertTrue(np.all(result >= 0))

    # Tests handling of zero intensity input
    def test_gaussian_zero_intensity(self):
        result = gaussian(np.linspace(0, 10, 5), 5, 0, 1)
        self.assertTrue(np.all(result == 0))

    # Tests error handling for invalid sigma values
    def test_gaussian_zero_sigma(self):
        with self.assertRaises(ZeroDivisionError):
            gaussian(0, 0, 1, 0)

    # --- Spectrum Reconstruction Tests ---

    # Tests reconstruction of spectrum from peak parameters
    def test_reconstruct_spectrum(self):
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, SAMPLE_PEAKS)
        self.assertEqual(result.shape, x_axis.shape)
        self.assertGreater(np.max(result), 0.5)

    # Tests spectrum reconstruction with empty peak list
    def test_reconstruct_spectrum_empty_peaks(self):
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, [])
        self.assertTrue(np.all(result == 0))

    # --- Spectrum Combination Tests ---
    """"
    # Tests combination of spectra with duplicate components
    def test_combine_spectra_from_peaks(self):
        components = ["Isocyanide", "Isocyanide"]
        x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, components)
        peak_idx = np.abs(x - 2100).argmin()
        self.assertLess(transmittance[peak_idx], 0.2)

    # Tests spectrum combination with empty component list
    def test_combine_spectra_from_peaks_empty_components(self):
        x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, [])
        self.assertTrue(np.all(transmittance == 1))

    # Tests handling of unknown components in spectrum combination
    def test_combine_spectra_from_peaks_unknown_components(self):
        x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, ["Unknown"])
        self.assertTrue(np.all(transmittance == 1))

    # Tests custom x-axis handling in spectrum combination
    def test_combine_spectra_from_peaks_custom_axis(self):
        custom_x = np.linspace(500, 3000, 100)
        x, _ = combine_spectra_from_peaks(SAMPLE_SPECTRA, ["Isocyanide"], custom_x)
        self.assertTrue(np.array_equal(x, custom_x))
    """
    # --- Property-Based Tests ---

    # Tests Gaussian intensity scaling with different input values
    def test_gaussian_intensity_scaling(self):
        for intensity in [0.1, 0.5, 1.0]:
            with self.subTest(intensity=intensity):
                x = np.linspace(0, 10, 100)
                result = gaussian(x, 5, intensity, 1)
                self.assertAlmostEqual(result.max(), intensity)

    # Tests component counting with different component quantities
    def test_combine_spectra_from_peaks_component_counting(self):
        for count in [1, 2, 5]:
            with self.subTest(count=count):
                components = ["Isocyanide"] * count
                x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, components)
                peak_idx = np.abs(x - 2100).argmin()
                expected = 1 - min(1, count * 0.9)
                self.assertAlmostEqual(transmittance[peak_idx], expected, delta=0.1)

    # --- Data Loading Tests ---

    # Tests JSON data loading functionality
    def test_functional_groups_json_loading(self):
        json_path = os.path.join(os.path.dirname(__file__), "../data/functional_groups_ir.json")
        if not os.path.exists(json_path):
            self.skipTest("JSON test data not available")
        
        with open(json_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("Isocyanide", data)

    # Tests module-based data loading fallback
    def test_dictionary_module_loading(self):
        module_path = os.path.join(os.path.dirname(__file__), "../../data/dictionnary.py")
        if not os.path.exists(module_path):
            self.skipTest("Module import test data not available")
        
        import importlib.util
        spec = importlib.util.spec_from_file_location("dictionnary", module_path)
        dictionnary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dictionnary)
        self.assertTrue(hasattr(dictionnary, 'FUNCTIONAL_GROUPS_IR'))


if __name__ == '__main__':
    unittest.main()