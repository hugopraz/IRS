import os
import json
import unittest
import numpy as np
from src.irs.ir_Structure import (
    gaussian,
    reconstruct_spectrum,
    validate_smiles,
    get_functional_groups,
    detect_main_functional_groups,
    count_ch_bonds,
    count_carbon_bonds_and_cn,
    analyze_molecule,
    FUNCTIONAL_GROUPS,
)


# Test data constants
SAMPLE_PEAKS = [(1000, 0.8, 50), (1500, 0.5, 30)]
SAMPLE_SPECTRA = {
    "Isocyanide": [(2100, 0.9, 40), (1200, 0.3, 20)],
    "Hydroxyl": [(3400, 0.7, 60)]
}

class TestIRStructureFunctions(unittest.TestCase):
    
    def test_validate_smiles_valid(self):
        valid_smiles = ["C", "CC", "C=C", "c1ccccc1", "CC(=O)O", "CCO"]
        for smiles in valid_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))
    
    def test_validate_smiles_invalid(self):
        invalid_smiles = ["X", "C(", "Si"]
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError):
                    validate_smiles(smiles)
    
    def test_validate_smiles_disallowed_atoms(self):
        disallowed_atoms = ["CSi", "CP", "CZn"]
        for smiles in disallowed_atoms:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    validate_smiles(smiles)
                self.assertIn("not allowed", str(context.exception))
    
    def test_validate_smiles_charged_atoms(self):
        charged_smiles = ["C[N+]", "C[O-]"]
        for smiles in charged_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    validate_smiles(smiles)
                self.assertIn("Charged atom", str(context.exception))
    
    def test_validate_smiles_aromatic_rings(self):
        valid_ring = "c1ccccc1"
        invalid_ring = "c1ncccc1N"
        
        self.assertTrue(validate_smiles(valid_ring))
        with self.assertRaises(ValueError) as context:
            validate_smiles(invalid_ring)
        self.assertIn("Aromatic ring", str(context.exception))
    
    def test_get_functional_groups(self):
        smiles = "CC(=O)O"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Carboxylic Acid", result)
        self.assertGreater(result["Carboxylic Acid"], 0)
    
    def test_get_functional_groups_empty(self):
        smiles = "C"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertEqual(len(result), 0)
    
    def test_get_functional_groups_pyridine(self):
        smiles = "c1ccncc1"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Pyridine", result)
        self.assertEqual(result["Pyridine"], 1)
    
    def test_detect_main_functional_groups(self):
        smiles = "c1ccc2ccccc2c1"
        result = detect_main_functional_groups(smiles)
        self.assertIn("Naphthalene", result)
        self.assertNotIn("Benzene", result)
        smiles = "CC(=O)OC"
        result = detect_main_functional_groups(smiles)
        self.assertIn("Ester", result)
        self.assertNotIn("Ketone", result)
    
    def test_detect_main_functional_groups_priority(self):
        smiles = "C1=CC=C2C(=C1)C=CN2"
        result = detect_main_functional_groups(smiles)
        self.assertIn("Indole", result)
        self.assertNotIn("Benzene", result)
        self.assertNotIn("Pyrrole", result)
    
    def test_count_ch_bonds(self):
        smiles = "C=CC#C"
        result = count_ch_bonds(smiles)
        self.assertIn("sp³ C-H", result)
        self.assertIn("sp² C-H", result)
        self.assertIn("sp C-H", result)
        self.assertGreater(result["sp³ C-H"], 0)
        self.assertGreater(result["sp² C-H"], 0)
        self.assertGreater(result["sp C-H"], 0)
    
    def test_count_ch_bonds_methane(self):
        smiles = "C"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 4)
        self.assertEqual(result["sp² C-H"], 0)
        self.assertEqual(result["sp C-H"], 0)
    
    def test_count_ch_bonds_benzene(self):
        smiles = "c1ccccc1"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 0)
        self.assertEqual(result["sp² C-H"], 6)
        self.assertEqual(result["sp C-H"], 0)
    
    def test_count_ch_bonds_acetylene(self):
        smiles = "C#C"
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp³ C-H"], 0)
        self.assertEqual(result["sp² C-H"], 0)
        self.assertEqual(result["sp C-H"], 2)
    
    def test_count_carbon_bonds_and_cn(self):
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
        smiles = "c1ccccc1"
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C=C (double)"], 6)  
        self.assertEqual(result["C–C (single)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
    
    def test_count_carbon_bonds_ethane(self):
        smiles = "CC"
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C–C (single)"], 1)
        self.assertEqual(result["C=C (double)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
        self.assertEqual(result["C–N (single)"], 0)
    
    def test_analyze_molecule(self):
        smiles = "CC(=O)O"  
        result = analyze_molecule(smiles)
        self.assertIsInstance(result, dict)
        self.assertIn("Carboxylic Acid", result)
        self.assertIn("sp³ C-H", result)
        self.assertIn("C–C (single)", result)
    
    def test_analyze_molecule_invalid(self):
        with self.assertRaises(ValueError):
            analyze_molecule("X")
    
    def test_analyze_molecule_integration(self):
        smiles = "c1ccccc1CC(=O)N"  
        result = analyze_molecule(smiles)
        self.assertIn("Benzene", result)
        self.assertIn("Amide", result)
        self.assertIn("sp² C-H", result)
        self.assertIn("sp³ C-H", result)
        self.assertNotIn("Ester", result)

    def test_gaussian(self):
        x = np.array([0, 1, 2])
        result = gaussian(x, center=1, intensity=1.0, sigma=0.5)
        self.assertAlmostEqual(result[1], 1.0)
        self.assertEqual(result[0], result[2])
        self.assertTrue(np.all(result >= 0))

    def test_gaussian_zero_intensity(self):
        result = gaussian(np.linspace(0, 10, 5), 5, 0, 1)
        self.assertTrue(np.all(result == 0))

    def test_gaussian_zero_sigma(self):
        with self.assertRaises(ZeroDivisionError):
            gaussian(0, 0, 1, 0)

    def test_reconstruct_spectrum(self):
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, SAMPLE_PEAKS)
        self.assertEqual(result.shape, x_axis.shape)
        self.assertGreater(np.max(result), 0.5)

    def test_reconstruct_spectrum_empty_peaks(self):
        x_axis = np.linspace(400, 4000, 5000)
        result = reconstruct_spectrum(x_axis, [])
        self.assertTrue(np.all(result == 0))

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

    def test_gaussian_intensity_scaling(self):
        for intensity in [0.1, 0.5, 1.0]:
            with self.subTest(intensity=intensity):
                x = np.linspace(0, 10, 100)
                result = gaussian(x, 5, intensity, 1)
                self.assertAlmostEqual(result.max(), intensity)

    def test_combine_spectra_from_peaks_component_counting(self):
        for count in [1, 2, 5]:
            with self.subTest(count=count):
                components = ["Isocyanide"] * count
                x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, components)
                peak_idx = np.abs(x - 2100).argmin()
                expected = 1 - min(1, count * 0.9)
                self.assertAlmostEqual(transmittance[peak_idx], expected, delta=0.1)

    def test_functional_groups_json_loading(self):
        json_path = os.path.join(os.path.dirname(__file__), "../data/functional_groups_ir.json")
        if not os.path.exists(json_path):
            self.skipTest("JSON test data not available")
        
        with open(json_path) as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        self.assertIn("Isocyanide", data)

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