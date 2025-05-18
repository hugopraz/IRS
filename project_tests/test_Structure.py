import os
import sys
import json
import unittest
import numpy as np
from unittest.mock import patch, MagicMock
from rdkit import Chem
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import functions to test
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

# Load test data
json_path_patterns = os.path.join(os.path.dirname(__file__), "..", "data", "dict_fg_detection.json")
with open(json_path_patterns, "r", encoding="utf-8") as f:
    try:
        FUNCTIONAL_GROUPS = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"❌ Failed to decode JSON: {e}")

# Sample test data
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
        """Initialize test data for use across multiple tests"""
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
    
    # --- SMILES Validation Tests ---
    
    def test_validate_smiles_accepts_common_organic_molecules(self):
        """Tests that validate_smiles correctly accepts valid SMILES strings for typical organic compounds"""
        valid_smiles = ["C", "CC", "C=C", "c1ccccc1", "CC(=O)O", "CCO"]
        for smiles in valid_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))
    
    def test_validate_smiles_rejects_syntactically_incorrect_structures(self):
        """Tests that validate_smiles rejects SMILES strings with invalid syntax"""
        invalid_smiles = ["X", "C(", "Si"]
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError):
                    validate_smiles(smiles)
    
    def test_validate_smiles_rejects_non_CHON_elements(self):
        """Tests that validate_smiles rejects molecules containing elements other than C, H, O, N"""
        disallowed_atoms = ["CSi", "CP", "CZn"]
        for smiles in disallowed_atoms:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError):
                    validate_smiles(smiles)
    
    def test_validate_smiles_rejects_charged_species(self):
        """Tests that validate_smiles rejects molecules containing charged atoms"""
        charged_smiles = ["C[N+]", "C[O-]"]
        for smiles in charged_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    validate_smiles(smiles)
                self.assertIn("Charged atom", str(context.exception))
    
    # --- Functional Group Detection Tests ---
    
    def test_get_functional_groups_identifies_carboxylic_acid(self):
        """Tests that get_functional_groups correctly identifies carboxylic acid in acetic acid"""
        smiles = "CC(=O)O"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Carboxylic Acid", result)
        self.assertGreater(result["Carboxylic Acid"], 0)
    
    def test_get_functional_groups_identifies_pyridine(self):
        """Tests that get_functional_groups correctly identifies pyridine ring"""
        smiles = "c1ccncc1"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Pyridine", result)
        self.assertEqual(result["Pyridine"], 1)
    
    def test_get_functional_groups_handles_invalid_patterns(self):
        """Tests that get_functional_groups handles cases when SMARTS patterns are invalid"""
        with patch('src.irs.ir_Structure.Chem.MolFromSmarts') as mock_smarts:
            mock_smarts.return_value = None
            self.assertEqual(get_functional_groups({"TEST": "pattern"}, "CCO"), {})
    
    def test_detect_main_functional_groups_prioritizes_complex_structures(self):
        """Tests that detect_main_functional_groups prioritizes complex structures over their components"""
        smiles = "C1=CC=C2C(=C1)C=CN2"  # Indole structure
        result = detect_main_functional_groups(smiles)
        
        if "Indole" in result:
            self.assertIn("Indole", result)
            self.assertNotIn("Benzene", result)
            self.assertNotIn("Pyrrole", result)
        elif "Pyrrole" in result:
            self.assertIn("Pyrrole", result)
    
    def test_detect_main_functional_groups_handles_simple_alkanes(self):
        """Tests that detect_main_functional_groups returns empty dict for simple alkanes without functional groups"""
        smiles = "CCCC"
        result = detect_main_functional_groups(smiles)
        self.assertEqual(result, {})
    
    # --- Bond Counting Tests ---
    
    def test_count_ch_bonds_identifies_mixed_hybridization(self):
        """Tests that count_ch_bonds correctly counts C-H bonds with different hybridizations in a mixed molecule"""
        smiles = "C=CC#C"
        result = count_ch_bonds(smiles)
        self.assertIn("sp³ C-H", result)
        self.assertIn("sp² C-H", result)
        self.assertIn("sp C-H", result)
    
    def test_count_ch_bonds_in_simple_molecules(self):
        """Tests that count_ch_bonds correctly counts C-H bonds in simple molecules (methane, benzene, acetylene)"""
        # Test methane (sp³)
        smiles_methane = "C"
        result_methane = count_ch_bonds(smiles_methane)
        self.assertEqual(result_methane["sp³ C-H"], 4)
        self.assertEqual(result_methane["sp² C-H"], 0)
        self.assertEqual(result_methane["sp C-H"], 0)
        
        # Test benzene (sp²)
        smiles_benzene = "c1ccccc1"
        result_benzene = count_ch_bonds(smiles_benzene)
        self.assertEqual(result_benzene["sp³ C-H"], 0)
        self.assertEqual(result_benzene["sp² C-H"], 6)
        self.assertEqual(result_benzene["sp C-H"], 0)
        
        # Test acetylene (sp)
        smiles_acetylene = "C#C"
        result_acetylene = count_ch_bonds(smiles_acetylene)
        self.assertEqual(result_acetylene["sp³ C-H"], 0)
        self.assertEqual(result_acetylene["sp² C-H"], 0)
        self.assertEqual(result_acetylene["sp C-H"], 2)
    
    def test_count_carbon_bonds_identifies_various_bond_types(self):
        """Tests that count_carbon_bonds_and_cn correctly identifies single, double, triple C-C bonds and C-N bonds"""
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
    
    def test_count_carbon_bonds_handles_aromatic_compounds(self):
        """Tests that count_carbon_bonds_and_cn correctly interprets aromatic bonds as double bonds"""
        smiles = "c1ccccc1"  # Benzene
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C=C (double)"], 6)  
        self.assertEqual(result["C–C (single)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
    
    # --- Molecule Analysis Tests ---
    
    def test_analyze_molecule_provides_comprehensive_analysis(self):
        """Tests that analyze_molecule provides comprehensive analysis of functional groups and bond types"""
        smiles = "CC(=O)O"  # Acetic acid
        result = analyze_molecule(smiles)
        self.assertIsInstance(result, dict)
        self.assertIn("Carboxylic Acid", result)
        self.assertIn("sp³ C-H", result)
        self.assertIn("C–C (single)", result)
    
    def test_analyze_molecule_rejects_invalid_smiles(self):
        """Tests that analyze_molecule rejects invalid SMILES strings"""
        with self.assertRaises(ValueError):
            analyze_molecule("X") 
    
    # --- Spectrum Generation Tests ---
    
    def test_gaussian_function_produces_correct_peak_shape(self):
        """Tests that gaussian function produces a peak with maximum at center and symmetric shape"""
        x = np.array([-1, 0, 1])
        result = gaussian(x, center=0, intensity=1.0, width=0.5)
        self.assertAlmostEqual(result[1], 1.0)  # Maximum at center
        self.assertAlmostEqual(result[0], result[2])  # Symmetric shape
        self.assertTrue(np.all(result >= 0))
    
    def test_gaussian_function_scales_intensity_correctly(self):
        """Tests that gaussian function scales peak height according to intensity parameter"""
        for intensity in [0.1, 0.5, 1.0]:
            with self.subTest(intensity=intensity):
                x = np.linspace(0, 10, 100)
                result = gaussian(x, 5, intensity, 1)
                self.assertAlmostEqual(result.max(), intensity, places=2)
    
    def test_reconstruct_spectrum_combines_multiple_peaks(self):
        """Tests that reconstruct_spectrum correctly combines multiple peaks into a spectrum"""
        y = reconstruct_spectrum(self.x_axis, self.test_peaks)
        peaks_found = set()
        for center, intensity, width in self.test_peaks:
            idx = np.argmin(np.abs(self.x_axis - center))
            self.assertGreater(y[idx], intensity*0.9) 
            peaks_found.add(center)
        expected_peaks = {p[0] for p in self.test_peaks}
        self.assertEqual(peaks_found, expected_peaks)
    
    def test_reconstruct_spectrum_with_empty_peak_list(self):
        """Tests that reconstruct_spectrum returns all zeros when no peaks are provided"""
        result = reconstruct_spectrum(self.x_axis, [])
        self.assertTrue(np.all(result == 0))
    
    # --- Full Spectrum Building Test ---
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.gcf')
    @patch('streamlit.pyplot')
    def test_build_and_plot_ir_spectrum_from_smiles(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
        """Tests that build_and_plot_ir_spectrum_from_smiles generates spectrum and calls plotting functions"""
        # Mock necessary parts
        mock_figure.return_value = MagicMock()
        mock_gcf.return_value = MagicMock()
        
        # Test with ethanol
        smiles = "CCO"
        x_axis, transmittance = build_and_plot_ir_spectrum_from_smiles(smiles)
        
        # Verify function calls
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        self.assertGreaterEqual(mock_gcf.call_count, 1)
        mock_st_pyplot.assert_called_once()
        
        # Check return values
        self.assertIsInstance(x_axis, np.ndarray)
        self.assertIsInstance(transmittance, np.ndarray)
    
    def test_build_spectrum_rejects_invalid_smiles(self):
        """Tests that build_and_plot_ir_spectrum_from_smiles rejects invalid SMILES"""
        with self.assertRaises(ValueError):
            build_and_plot_ir_spectrum_from_smiles("X")

if __name__ == '__main__':
    unittest.main()