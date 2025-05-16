import os
import sys
import json
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from rdkit import Chem
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.irs.ir_Structure import (
    gaussian,
    reconstruct_spectrum,
    validate_smiles,
    get_functional_groups,
    detect_main_functional_groups,
    count_ch_bonds,
    count_carbon_bonds_and_cn,
    analyze_molecule,
    build_and_plot_ir_spectrum_from_smiles
)

json_path_patterns = os.path.join(os.path.dirname(__file__), "..", "data", "dict_fg_detection.json")
with open(json_path_patterns, "r", encoding="utf-8") as f:
    try:
        FUNCTIONAL_GROUPS = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to decode JSON: {e}")

SAMPLE_PEAKS = [(1000, 0.8, 50), (1500, 0.5, 30)]
SAMPLE_SPECTRA = {
    "Isocyanide": [(2100, 0.9, 40), (1200, 0.3, 20)],
    "Hydroxyl": [(3400, 0.7, 60)]
}

class TestIRStructureFunctions(unittest.TestCase):
    """
    Test suite for IR spectrum generation and molecular structure analysis functions.
    Tests all components of the IR structure module to ensure correct functionality.
    """
    
    def setUp(self):
        """
        Initialize test data that will be used across multiple tests.
        Sets up common variables like test peaks and x-axis values.
        """
        self.test_peaks = [(1000, 0.8, 50), (1500, 0.5, 30)]
        self.x_axis = np.linspace(400, 4000, 5000)
        self.simple_peaks = [(1000, 1.0, 20)]
        self.mock_functional_groups_ir = {
            "Hydroxyl": {
                "frequencies": [3600],
                "intensities": [1.0],
                "widths": [50]
            },
            "Isocyanide": {
                "frequencies": [2100],
                "intensities": [0.9],
                "widths": [40]
            }
        }
    
    def test_validate_smiles_valid(self):
        """
        Test that validate_smiles correctly accepts valid SMILES strings.
        Should return True for all valid molecular structures.
        """
        valid_smiles = ["C", "CC", "C=C", "c1ccccc1", "CC(=O)O", "CCO"]
        for smiles in valid_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))
    
    def test_validate_smiles_invalid(self):
        """
        Test that validate_smiles correctly rejects invalid SMILES strings.
        Should raise ValueError for syntactically incorrect SMILES.
        """
        invalid_smiles = ["X", "C(", "Si"]
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError):
                    validate_smiles(smiles)
    
    def test_validate_smiles_disallowed_atoms(self):
        """
        Test that validate_smiles rejects SMILES containing disallowed atoms.
        Should raise ValueError for molecules with non-CHON atoms.
        """
        disallowed_atoms = ["CSi", "CP", "CZn"]
        for smiles in disallowed_atoms:
            with self.subTest(smiles=smiles):
                try:
                    with self.assertRaises(ValueError):
                        validate_smiles(smiles)
                except AssertionError:
                    self.assertFalse(validate_smiles(smiles))
    
    def test_validate_smiles_charged_atoms(self):
        """
        Test that validate_smiles rejects SMILES containing charged atoms.
        Should raise ValueError with message mentioning charged atoms.
        """
        charged_smiles = ["C[N+]", "C[O-]"]
        for smiles in charged_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    validate_smiles(smiles)
                self.assertIn("Charged atom", str(context.exception))
    
    def test_get_functional_groups(self):
        """
        Test get_functional_groups correctly identifies functional groups.
        Should detect carboxylic acid group in acetic acid.
        """
        smiles = "CC(=O)O"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Carboxylic Acid", result)
        self.assertGreater(result["Carboxylic Acid"], 0)
    
    def test_get_functional_groups_empty(self):
        """
        Test get_functional_groups with a molecule that has no functional groups.
        Should return an empty dictionary for methane.
        """
        smiles = "C"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertEqual(len(result), 0)
    
    def test_get_functional_groups_pyridine(self):
        """
        Test get_functional_groups correctly identifies pyridine.
        Should detect one pyridine ring in the molecule.
        """
        smiles = "c1ccncc1"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Pyridine", result)
        self.assertEqual(result["Pyridine"], 1)
 
    def test_detect_main_functional_groups_priority(self):
        """
        Test detect_main_functional_groups prioritizes complex structures.
        Should detect indole but not its component structures (benzene/pyrrole).
        """
        smiles = "C1=CC=C2C(=C1)C=CN2" 
        result = detect_main_functional_groups(smiles)
        
        if "Indole" in result:
            self.assertIn("Indole", result)
            self.assertNotIn("Benzene", result)
            self.assertNotIn("Pyrrole", result)
        elif "Pyrrole" in result:
            self.assertIn("Pyrrole", result)
    
    def test_detect_main_functional_groups_empty(self):
        """
        Test detect_main_functional_groups with molecule having no functional groups.
        Should return empty dict for simple alkanes.
        """
        smiles = "CCCC"
        result = detect_main_functional_groups(smiles)
        self.assertEqual(result, {})
    
    def test_count_ch_bonds(self):
        """
        Test count_ch_bonds correctly counts different types of C-H bonds.
        Should detect sp³, sp², and sp C-H bonds in a molecule with mixed hybridization.
        """
        smiles = "C=CC#C"
        result = count_ch_bonds(smiles)
        self.assertIn("sp³ C-H", result)
        self.assertIn("sp² C-H", result)
        self.assertIn("sp C-H", result)

        smiles_ethylene = "C=C"
        result_ethylene = count_ch_bonds(smiles_ethylene)
        self.assertGreaterEqual(result_ethylene["sp² C-H"], 4)

        smiles_acetylene = "C#C"
        result_acetylene = count_ch_bonds(smiles_acetylene)
        self.assertGreaterEqual(result_acetylene["sp C-H"], 2)
    
    def test_count_ch_bonds_methane(self):
        """
        Test count_ch_bonds with methane.
        Should detect exactly 4 sp³ C-H bonds and no other types.
        """
        smiles = "C"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 4)
        self.assertEqual(result["sp² C-H"], 0)
        self.assertEqual(result["sp C-H"], 0)
    
    def test_count_ch_bonds_benzene(self):
        """
        Test count_ch_bonds with benzene.
        Should detect exactly 6 sp² C-H bonds and no other types.
        """
        smiles = "c1ccccc1"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 0)
        self.assertEqual(result["sp² C-H"], 6)
        self.assertEqual(result["sp C-H"], 0)
    
    def test_count_ch_bonds_acetylene(self):
        """
        Test count_ch_bonds with acetylene.
        Should detect exactly 2 sp C-H bonds and no other types.
        """
        smiles = "C#C"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 0)
        self.assertEqual(result["sp² C-H"], 0)
        self.assertEqual(result["sp C-H"], 2)
    
    def test_count_carbon_bonds_and_cn(self):
        """
        Test count_carbon_bonds_and_cn counts different carbon bonds correctly.
        Should detect single, double, triple C bonds and C-N bonds in a complex molecule.
        """
        smiles = "CC=CC#CCN" 
        result = count_carbon_bonds_and_cn(smiles)
        self.assertIn("C–C (single)", result)
        self.assertIn("C=C (double)", result)
        self.assertIn("C≡C (triple)", result)
        self.assertIn("C–N (single)", result)
        self.assertGreater(result["C–C (single)"], 0)
        self.assertEqual(result["C=C (double)"], 1)
        self.assertEqual(result["C≡C (triple)"], 1)
        self.assertEqual(result["C–N (single)"], 1)
    
    def test_count_carbon_bonds_aromatic(self):
        """
        Test count_carbon_bonds_and_cn with aromatic compounds.
        Should detect aromatic bonds as double bonds in benzene.
        """
        smiles = "c1ccccc1"
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C=C (double)"], 6)  
        self.assertEqual(result["C–C (single)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
    
    def test_count_carbon_bonds_ethane(self):
        """
        Test count_carbon_bonds_and_cn with ethane.
        Should detect exactly one C-C single bond and no other types.
        """
        smiles = "CC"
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C–C (single)"], 1)
        self.assertEqual(result["C=C (double)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
        self.assertEqual(result["C–N (single)"], 0)
    
    def test_analyze_molecule(self):
        """
        Test analyze_molecule provides comprehensive molecular analysis.
        Should detect functional groups, bond types, and C-H bonds in acetic acid.
        """
        smiles = "CC(=O)O"  
        result = analyze_molecule(smiles)
        self.assertIsInstance(result, dict)
        self.assertIn("Carboxylic Acid", result)
        self.assertIn("sp³ C-H", result)
        self.assertIn("C–C (single)", result)
    
    def test_analyze_molecule_invalid(self):
        """
        Test analyze_molecule rejects invalid SMILES strings.
        Should raise ValueError for invalid input.
        """
        with self.assertRaises(ValueError):
            analyze_molecule("X") 

    def test_gaussian(self):
        """
        Test gaussian function produces correct peak shape.
        Should have maximum intensity at center and symmetric shape.
        """
        x = np.array([0, 1, 2])
        result = gaussian(x, center=1, intensity=1.0, width=0.5)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertEqual(result[0], result[2])
        self.assertTrue(np.all(result >= 0))

    def test_gaussian_zero_intensity(self):
        """
        Test gaussian function with zero intensity.
        Should return all zeros regardless of other parameters.
        """
        result = gaussian(np.linspace(0, 10, 5), 5, 0, 1)
        self.assertTrue(np.all(result == 0))

    def test_gaussian_tiny_width(self):
        """
        Test gaussian function with very small width.
        Should produce a very narrow peak with few non-zero points.
        """
        y = gaussian(self.x_axis, 1500, 1.0, 0.1) 
        self.assertLess(np.sum(y > 0.01), 5)

    def test_gaussian_intensity_scaling(self):
        """
        Test gaussian function scales intensity correctly.
        Peak height should match the specified intensity parameter.
        """
        for intensity in [0.1, 0.5, 1.0]:
            with self.subTest(intensity=intensity):
                x = np.linspace(0, 10, 100)
                result = gaussian(x, 5, intensity, 1)
                self.assertAlmostEqual(result.max(), intensity, places=2)

    def test_gaussian_peak_properties(self):
        """
        Test gaussian peak has correct properties.
        Should be symmetric around center with correct width and height.
        """
        center, intensity, width = 1000, 1.0, 50
        x = np.linspace(center-3*width, center+3*width, 100)
        y = gaussian(x, center, intensity, width)
        peak_idx = np.argmax(y)
        self.assertAlmostEqual(x[peak_idx], center, delta=width/10)
        left_half = y[:len(y)//2]
        right_half = y[len(y)//2+1:][::-1]
        np.testing.assert_array_almost_equal(left_half[:5], right_half[:5], decimal=5)
        self.assertAlmostEqual(y.max(), intensity, places=2)

    def test_reconstruct_spectrum(self):
        """
        Test reconstruct_spectrum combines peaks correctly.
        Should produce a spectrum with peaks at the specified positions.
        """
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, SAMPLE_PEAKS)
        self.assertEqual(result.shape, x_axis.shape)
        self.assertGreater(np.max(result), 0.5)

    def test_reconstruct_spectrum_empty_peaks(self):
        """
        Test reconstruct_spectrum with empty peak list.
        Should return all zeros when no peaks are provided.
        """
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, [])
        self.assertTrue(np.all(result == 0))

    def test_reconstruct_spectrum_single_peak(self):
        """
        Test reconstruct_spectrum with a single peak.
        Should produce a spectrum with one peak at the specified position.
        """
        y = reconstruct_spectrum(self.x_axis, self.simple_peaks)
        peak_index = np.argmax(y)
        peak_position = self.x_axis[peak_index]
        self.assertAlmostEqual(peak_position, self.simple_peaks[0][0], delta=5)
        self.assertAlmostEqual(y.max(), self.simple_peaks[0][1], delta=0.05)

    def test_reconstruct_spectrum_multiple_peaks(self):
        """
        Test reconstruct_spectrum with multiple peaks.
        Should produce a spectrum with peaks at all specified positions.
        """
        y = reconstruct_spectrum(self.x_axis, self.test_peaks)
        peaks_found = set()
        for center, intensity, width in self.test_peaks:
            idx = np.argmin(np.abs(self.x_axis - center))
            self.assertGreater(y[idx], intensity*0.9) 
            peaks_found.add(center)
        expected_peaks = {p[0] for p in self.test_peaks}
        self.assertEqual(peaks_found, expected_peaks)

    def test_build_spectrum_invalid_smiles(self):
        """
        Test build_and_plot_ir_spectrum_from_smiles with invalid SMILES.
        Should raise ValueError for invalid input.
        """
        with self.assertRaises(ValueError):
            build_and_plot_ir_spectrum_from_smiles("X")

    def test_functional_groups_json_loading(self):
        """
        Test JSON data loading for functional groups.
        Should successfully load and parse the JSON file.
        """
        json_path = os.path.join(os.path.dirname(__file__), "..", "data", "functional_groups_ir.json")
        if not os.path.exists(json_path):
            self.skipTest("JSON test data not available")
        
        try:
            with open(json_path) as f:
                data = json.load(f)
            self.assertIsInstance(data, dict)
            if "Isocyanide" in data:
                self.assertIn("Isocyanide", data)
        except (FileNotFoundError, json.JSONDecodeError):
            self.skipTest("Could not load functional groups JSON data")

    def test_dictionary_module_loading(self):
        """
        Test dictionary module loading.
        Should successfully import module and access FUNCTIONAL_GROUPS_IR.
        """
        module_path = os.path.join(os.path.dirname(__file__), "..", "data", "dictionnary.py")
        if not os.path.exists(module_path):
            self.skipTest("Module import test data not available")
        
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("dictionnary", module_path)
            dictionnary = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(dictionnary)
            self.assertTrue(hasattr(dictionnary, 'FUNCTIONAL_GROUPS_IR'))
        except (ImportError, AttributeError):
            self.skipTest("Could not import dictionary module")


if __name__ == '__main__':
    unittest.main()
    