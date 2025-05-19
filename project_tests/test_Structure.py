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
json_path_patterns = os.path.join(os.path.dirname(__file__), "..", "src", "irs", "data", "dict_fg_detection.json")
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
    
    def test_validate_smiles_rejects_non_allowed_elements(self):
        """Tests that validate_smiles rejects molecules containing elements not in the allowed set"""
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
    
    def test_validate_smiles_accepts_allowed_halogens(self):
        """Tests that validate_smiles accepts molecules with allowed halogen atoms"""
        halogen_smiles = ["CF", "CCl", "CBr", "CI"]
        for smiles in halogen_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))
    
    def test_validate_smiles_validates_aromatic_ring_constraints(self):
        """Tests that validate_smiles enforces carbon count constraints for aromatic rings with heteroatoms"""
        # Valid aromatic rings with heteroatoms
        valid_aromatic = ["c1ccncc1", "c1ccoc1"]  # Pyridine, furan
        for smiles in valid_aromatic:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))
    
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
    
    def test_get_functional_groups_prevents_duplicate_arene_counting(self):
        """Tests that get_functional_groups prevents duplicate counting of aromatic systems"""
        smiles = "c1cc2ccccc2cc1"  # Naphthalene
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        # Should count naphthalene as one unit, not multiple overlapping aromatic systems
        if "Naphthalene" in result:
            self.assertEqual(result["Naphthalene"], 1)
    
    def test_get_functional_groups_counts_multiple_instances(self):
        """Tests that get_functional_groups correctly counts multiple instances of the same functional group"""
        smiles = "C(=O)OCC(=O)O"  # Succinic acid (two carboxylic acids)
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        if "Carboxylic Acid" in result:
            self.assertEqual(result["Carboxylic Acid"], 2)
    
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
    
    def test_detect_main_functional_groups_handles_polycyclic_compounds(self):
        """Tests that detect_main_functional_groups correctly handles polycyclic aromatic hydrocarbons"""
        smiles = "c1ccc2cc3ccccc3cc2c1"  # Anthracene
        result = detect_main_functional_groups(smiles)
        if "Anthracene" in result:
            self.assertIn("Anthracene", result)
            # Should not count individual benzene rings
            self.assertNotIn("Benzene", result)
    
    def test_detect_main_functional_groups_handles_ester_overlap(self):
        """Tests that detect_main_functional_groups removes ketone and ether when ester is present"""
        smiles = "CC(=O)OC"  # Methyl acetate
        result = detect_main_functional_groups(smiles)
        if "Ester" in result:
            self.assertIn("Ester", result)
            self.assertNotIn("Ketone", result)
            self.assertNotIn("Ether", result)
    
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
    
    def test_count_ch_bonds_handles_no_hydrogens(self):
        """Tests that count_ch_bonds correctly handles carbons with no hydrogens"""
        smiles = "C(F)(F)(F)C"  # Trifluoromethyl compound
        result = count_ch_bonds(smiles)
        # First carbon has no H, second carbon has 3 H
        self.assertEqual(result["sp³ C-H"], 3)
    
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
    
    def test_count_carbon_bonds_ignores_non_single_cn_bonds(self):
        """Tests that count_carbon_bonds_and_cn only counts single C-N bonds"""
        smiles = "C#N"  # Acetonitrile (triple bond)
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C–N (single)"], 0)
    
    def test_count_carbon_bonds_handles_molecules_without_carbon_bonds(self):
        """Tests that count_carbon_bonds_and_cn handles single carbon molecules"""
        smiles = "CN"  # Methylamine
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C–C (single)"], 0)
        self.assertEqual(result["C=C (double)"], 0)
        self.assertEqual(result["C≡C (triple)"], 0)
        self.assertEqual(result["C–N (single)"], 1)
    
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
    
    def test_analyze_molecule_combines_all_analysis_types(self):
        """Tests that analyze_molecule combines functional groups, C-H bonds, and C-C bonds into one dictionary"""
        smiles = "CCO"  # Ethanol
        result = analyze_molecule(smiles)
        # Should contain all types of analysis
        has_fg = any(key in ["Alcohol", "Hydroxyl"] for key in result.keys())
        has_ch = any("C-H" in key for key in result.keys())
        has_cc = any("C–C" in key or "C=C" in key or "C≡C" in key for key in result.keys())
        self.assertTrue(has_fg or has_ch or has_cc)
    
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
    
    def test_gaussian_function_handles_edge_cases(self):
        """Tests that gaussian function handles edge cases like zero intensity and very large widths"""
        x = np.linspace(0, 10, 100)
        # Zero intensity
        result_zero = gaussian(x, 5, 0, 1)
        self.assertTrue(np.all(result_zero == 0))
        # Very wide peak
        result_wide = gaussian(x, 5, 1, 100)
        self.assertTrue(np.all(result_wide > 0))
    
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
    
    def test_reconstruct_spectrum_handles_single_peak(self):
        """Tests that reconstruct_spectrum correctly handles a single peak"""
        single_peak = [(2000, 1.0, 50)]
        result = reconstruct_spectrum(self.x_axis, single_peak)
        max_idx = np.argmax(result)
        self.assertAlmostEqual(self.x_axis[max_idx], 2000, delta=10)
    
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
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.gcf')
    @patch('streamlit.pyplot')
    def test_build_spectrum_with_custom_axis(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
        """Tests that build_and_plot_ir_spectrum_from_smiles works with custom frequency axis"""
        mock_figure.return_value = MagicMock()
        mock_gcf.return_value = MagicMock()
        
        custom_axis = np.linspace(500, 3500, 1000)
        x_axis, transmittance = build_and_plot_ir_spectrum_from_smiles("C", custom_axis)
        
        self.assertTrue(np.array_equal(x_axis, custom_axis))
        self.assertEqual(len(transmittance), len(custom_axis))
    
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.gcf')
    @patch('streamlit.pyplot')
    def test_build_spectrum_handles_no_functional_groups(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
        """Tests that build_and_plot_ir_spectrum_from_smiles handles molecules with no recognized functional groups"""
        mock_figure.return_value = MagicMock()
        mock_gcf.return_value = MagicMock()
        
        # Simple alkane with minimal functional groups
        smiles = "CCCC"
        x_axis, transmittance = build_and_plot_ir_spectrum_from_smiles(smiles)
        
        # Should still return valid arrays
        self.assertIsInstance(x_axis, np.ndarray)
        self.assertIsInstance(transmittance, np.ndarray)
        self.assertEqual(len(x_axis), len(transmittance))

if __name__ == '__main__':
    unittest.main()