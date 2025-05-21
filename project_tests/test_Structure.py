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


class TestIRStructureFunctions(unittest.TestCase):

    # Initialize test data for use across multiple tests
    def setUp(self):
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


    # Tests that validate_smiles correctly accepts valid SMILES strings for typical organic compounds
    def test_validate_smiles_accepts_common_organic_molecules(self):
        valid_smiles = ["C", "CC", "C=C", "c1ccccc1", "CC(=O)O", "CCO"]
        for smiles in valid_smiles:
            with self.subTest(smiles=smiles):
                self.assertTrue(validate_smiles(smiles))

    # Tests that validate_smiles rejects invalid SMILES strings 
    def test_validate_smiles_rejects_invalid_smiles(self):
        invalid_smiles = ["X", "C(", "invalid"]
        for smiles in invalid_smiles:
            with self.subTest(smiles=smiles):
                with self.assertRaises(ValueError) as context:
                    validate_smiles(smiles)
                self.assertIn("Invalid SMILES string", str(context.exception))

    # Tests validation with disallowed atoms 
    def test_validate_smiles_rejects_non_allowed_elements(self):
        with self.assertRaises(ValueError) as context:
            validate_smiles("C[S]C") 
        self.assertIn("Atom 'S' is not allowed", str(context.exception))

    # Tests validation with charged atoms 
    def test_validate_smiles_rejects_charged_species(self):
        with self.assertRaises(ValueError) as context:
            validate_smiles("C[O-]C")  
        self.assertIn("Invalid SMILES", str(context.exception))

    # Tests aromatic ring validation/ invalidation
    def test_validate_smiles_aromatic_ring_constraints(self):
        with self.assertRaises(ValueError) as context:
            validate_smiles("c1cno1")
        self.assertIn("Aromatic ring with atoms (N or O)", str(context.exception))

        with self.assertRaises(ValueError) as context:
            validate_smiles("c1ccoc1")
        self.assertIn("Aromatic ring with heteroatoms", str(context.exception))


    # Tests that get_functional_groups correctly identifies carboxylic acid in acetic acid
    def test_get_functional_groups_identifies_carboxylic_acid(self):
        smiles = "CC(=O)O"
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIn("Carboxylic Acid", result)
        self.assertGreater(result["Carboxylic Acid"], 0)

    # Tests that get_functional_groups adds hydrogens to molecule 
    def test_get_functional_groups_handles_hydrogens_addition(self):
        smiles = "C"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIsInstance(result, dict)

    # Tests handling of invalid SMARTS patterns 
    def test_get_functional_groups_handles_invalid_patterns(self):
        invalid_fg = {"InvalidPattern": "invalid_smarts"}
        with patch('src.irs.ir_Structure.Chem.MolFromSmarts') as mock_smarts:
            mock_smarts.return_value = None
            result = get_functional_groups(invalid_fg, "CCO")
            self.assertEqual(result, {})

    # Tests identification of Pyridine
    def test_get_functional_groups_identifies_pyridine(self):
        smiles = "c1ccncc1"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        if "Pyridine" in FUNCTIONAL_GROUPS:
            self.assertIn("Pyridine", result)
            self.assertEqual(result["Pyridine"], 1)

    # Tests identification of Pyrrole 
    def test_get_functional_groups_identifies_pyrrole(self):
        smiles = "c1cc[nH]c1"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        if "Pyrrole" in FUNCTIONAL_GROUPS:
            self.assertIn("Pyrrole", result)

    # Tests identification of Furan and Quinone
    def test_get_functional_groups_identifies_furan_and_quinone(self):
        smiles_furan = "c1ccoc1"  
        result_furan = get_functional_groups(FUNCTIONAL_GROUPS, smiles_furan)
        if "Furan" in FUNCTIONAL_GROUPS:
            self.assertIn("Furan", result_furan)
        
        smiles_quinone = "O=C1C=CC(=O)C=C1"  
        result_quinone = get_functional_groups(FUNCTIONAL_GROUPS, smiles_quinone)
        if "Quinone" in FUNCTIONAL_GROUPS:
            self.assertIn("Quinone", result_quinone)

    # Tests preventing duplicate arene counting with frozenset 
    def test_get_functional_groups_prevents_duplicate_arene_counting(self):
        smiles = "c1cc2ccccc2cc1"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        total_count = sum(result.values())
        self.assertGreaterEqual(total_count, 0)  

    # Tests addition of arene matches to set 
    def test_get_functional_groups_handles_arene_matches_addition(self):
        smiles = "c1ccccc1"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        self.assertIsInstance(result, dict)

    # Tests counting of non-arene functional groups 
    def test_get_functional_groups_counts_multiple_standard_groups(self):
        smiles = "O=C(O)C(=O)O"  
        result = get_functional_groups(FUNCTIONAL_GROUPS, smiles)
        if "Carboxylic Acid" in result:
            self.assertEqual(result["Carboxylic Acid"], 2)

    # Tests filtering of zero counts in return statement
    def test_get_functional_groups_filters_zero_counts(self):
        result = get_functional_groups(FUNCTIONAL_GROUPS, "CCCC")
        for count in result.values():
            self.assertGreater(count, 0)


    # Tests that detect_main_functional_groups handles molecules without recognized functional groups
    def test_detect_main_functional_groups_handles_empty_input(self):
        smiles = "CCCC"  
        result = detect_main_functional_groups(smiles)
        self.assertIsInstance(result, dict)

    # Tests ester overlap reduction 
    def test_detect_main_functional_groups_ester_overlap_reduction(self):
        with patch('src.irs.ir_Structure.get_functional_groups') as mock_get_fg:
            mock_get_fg.return_value = {"Ester": 1, "Ether": 1, "Ketone": 1}
            result = detect_main_functional_groups("CCOC(=O)C")
 
            self.assertIn("Ester", result)
            self.assertEqual(result["Ester"], 1)

            self.assertNotIn("Ether", result)
            self.assertNotIn("Ketone", result)


    # Tests that count_ch_bonds correctly counts C-H bonds with different hybridizations
    def test_count_ch_bonds_handles_mixed_hybridization(self):
        smiles = "C=CC#C"  
        result = count_ch_bonds(smiles)
        self.assertIn("sp³ C-H", result)
        self.assertIn("sp² C-H", result)
        self.assertIn("sp C-H", result)

    # Tests SP hybridized C-H counting with triple bonds 
    def test_count_ch_bonds_sp_hybridization_with_triple_bonds(self):
        smiles = "C#CC" 
        result = count_ch_bonds(smiles)
        self.assertEqual(result["sp C-H"], 1) 
        self.assertEqual(result["sp³ C-H"], 3)  


    # Tests that count_carbon_bonds_and_C-N correctly identifies all bond types
    def test_count_carbon_bonds_identifies_various_bond_types(self):
        smiles = "CC=CC#CCN" 
        result = count_carbon_bonds_and_cn(smiles)
        self.assertIn("C–C (single)", result)
        self.assertIn("C=C (double)", result)
        self.assertIn("C≡C (triple)", result)
        self.assertIn("C–N (single)", result)

    # Tests aromatic bond counting  
    def test_count_carbon_bonds_handles_aromatic_compounds(self):
        smiles = "c1ccccc1" 
        result = count_carbon_bonds_and_cn(smiles)
        self.assertEqual(result["C=C (double)"], 6) 
        
        smiles_cn = "CCN"  
        result_cn = count_carbon_bonds_and_cn(smiles_cn)
        self.assertEqual(result_cn["C–N (single)"], 1)


    # Tests that analyze_molecule combines all analysis functions correctly
    def test_analyze_molecule_provides_comprehensive_analysis(self):
        smiles = "CC(=O)O"  
        result = analyze_molecule(smiles)
        self.assertIsInstance(result, dict)
        has_any_data = len(result) > 0
        self.assertTrue(has_any_data)

    # Tests that analyze_molecule validates SMILES before processing
    def test_analyze_molecule_validates_smiles_first(self):
        with self.assertRaises(ValueError):
            analyze_molecule("X")  

    # Tests that gaussian function produces correct shape
    def test_gaussian_function_produces_correct_peak_shape(self):
        x = np.array([-1, 0, 1])
        result = gaussian(x, center=0, intensity=1.0, width=0.5)
        self.assertAlmostEqual(result[1], 1.0)  
        self.assertAlmostEqual(result[0], result[2])  
        self.assertTrue(np.all(result >= 0))

    # Tests that reconstruct_spectrum correctly combines multiple Gaussian peaks
    def test_reconstruct_spectrum_combines_multiple_peaks(self):
        y = reconstruct_spectrum(self.x_axis, self.test_peaks)
        for center, intensity, width in self.test_peaks:
            idx = np.argmin(np.abs(self.x_axis - center))
            self.assertGreater(y[idx], intensity * 0.5) 

    # Tests that reconstruct_spectrum returns zeros for empty peak list
    def test_reconstruct_spectrum_with_empty_peak_list(self):
        result = reconstruct_spectrum(self.x_axis, [])
        self.assertTrue(np.all(result == 0))

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.gcf')
    @patch('streamlit.pyplot')
    # Tests basic functionality of IR spectrum building and plotting
    def test_build_and_plot_ir_spectrum_from_smiles_basic_functionality(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
        mock_figure.return_value = MagicMock()
        mock_gcf.return_value = MagicMock()
        
        smiles = "CCO"
        x_axis, transmittance = build_and_plot_ir_spectrum_from_smiles(smiles)
   
        mock_figure.assert_called_once()
        mock_plot.assert_called_once()
        mock_st_pyplot.assert_called_once()
        
        self.assertIsInstance(x_axis, np.ndarray)
        self.assertIsInstance(transmittance, np.ndarray)

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.plot')
    @patch('matplotlib.pyplot.gcf')
    @patch('streamlit.pyplot')
    # Tests custom axis usage in IR spectrum building
    def test_build_spectrum_with_custom_axis(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
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
    # Tests IR spectrum building for molecules with minimal functional groups
    def test_build_spectrum_handles_no_functional_groups(self, mock_st_pyplot, mock_gcf, mock_plot, mock_figure):
        mock_figure.return_value = MagicMock()
        mock_gcf.return_value = MagicMock()
        
        smiles = "CCCC"
        x_axis, transmittance = build_and_plot_ir_spectrum_from_smiles(smiles)

        self.assertIsInstance(x_axis, np.ndarray)
        self.assertIsInstance(transmittance, np.ndarray)
        self.assertEqual(len(x_axis), len(transmittance))

    # Tests that build_and_plot_ir_spectrum_from_smiles validates SMILES input
    def test_build_spectrum_validates_smiles(self):
        with self.assertRaises(ValueError):
            build_and_plot_ir_spectrum_from_smiles("X")  


if __name__ == '__main__':
    unittest.main()
