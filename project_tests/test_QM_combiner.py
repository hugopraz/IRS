import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Path Configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# Import functions to test
from src.irs.QM_combiner import (
    generate_3d_molecule,
    guess_charge_multiplicity,
    write_orca_input,
    parse_orca_output,
    cleanup_orca_files,
    run_orca,
    plot_ir_spectrum,
    name_to_smiles,
    mol_to_3dviewer,
    psi4_calculate_frequencies,
    show_3dmol, 
    build_and_plot_ir_spectrum_from_smiles, 
    smiles_to_optimized_geometry
)

# Mock classes for testing
class MockMol:
    def __init__(self, charge=0, unpaired=0):
        self.charge = charge
        self.unpaired = unpaired
        self.atoms = []
        
    def GetFormalCharge(self):
        return self.charge
        
    def GetAtoms(self):
        return self.atoms
        
    def GetNumAtoms(self):
        return len(self.atoms)
        
    def GetConformer(self):
        return MockConformer()

class MockAtom:
    def __init__(self, symbol="C", idx=0, radical_electrons=0):
        self.symbol = symbol
        self.idx = idx
        self.radical_electrons = radical_electrons
        
    def GetSymbol(self):
        return self.symbol
        
    def GetIdx(self):
        return self.idx
        
    def GetNumRadicalElectrons(self):
        return self.radical_electrons

class MockConformer:
    def GetAtomPosition(self, idx):
        return MockPosition()

class MockPosition:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class TestOrcaFunctions(unittest.TestCase):

    # Create temp directory for tests
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        self.temp_dir.cleanup()

    # Verifies that a 3D molecule can be correctly generated from a valid SMILES string
    def test_generate_3d_from_smiles_returns_molecule(self):
        mol = generate_3d_molecule("CCO")
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumConformers(), 0)

    # Ensures that hydrogens are properly added to the molecular structure
    def test_generate_molecule_contains_hydrogens(self):
        mol = generate_3d_molecule("C")
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        self.assertIn("H", atoms)
    
    # Validates that the function properly handles invalid SMILES input by returning None
    def test_generate_3d_molecule_invalid_smiles(self):
        with patch('rdkit.Chem.MolFromSmiles', return_value=None):
            result = generate_3d_molecule("InvalidSMILES")
            self.assertIsNone(result)
    
    # Checks that the function correctly handles failures in 3D structure embedding
    def test_generate_3d_molecule_embedding_failure(self):
        mock_mol = MagicMock()
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=1):
            result = generate_3d_molecule("C")
            self.assertIsNone(result)
    
    # Ensures UFF optimization failures don't crash the function and a molecule is still returned
    def test_generate_3d_molecule_uff_failure(self):
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1
        
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=0), \
             patch('rdkit.Chem.AllChem.UFFOptimizeMolecule', side_effect=Exception("UFF failed")):
            result = generate_3d_molecule("C")
            self.assertIsNotNone(result)

    # Checks that neutral molecules are assigned charge 0 and singlet multiplicity 1
    def test_guess_charge_and_multiplicity_neutral(self):
        mol = generate_3d_molecule("CCO")
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 0)
        self.assertEqual(multiplicity, 1)

    # Verifies that radical species are correctly assigned doublet multiplicity (2)
    def test_guess_charge_and_multiplicity_radical(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
        AllChem.EmbedMolecule(mol)
        _, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(multiplicity, 2)
    
    # Confirms that positively charged molecules retain their formal charge
    def test_guess_charge_and_multiplicity_charged(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[N+](C)(C)C"))
        AllChem.EmbedMolecule(mol)
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 1)
        self.assertEqual(multiplicity, 1)

    # Verifies that ORCA input files are created with correct content and structure
    def test_write_orca_input_file_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mol = generate_3d_molecule("CO")
            inp_path = write_orca_input(mol, tmpdir, "co_test", "B3LYP def2-SVP", 0, 1)
            self.assertTrue(os.path.exists(inp_path))
            with open(inp_path) as f:
                content = f.read()
            self.assertIn("C", content)
            self.assertIn("O", content)
            self.assertIn("* xyz 0 1", content)
    
    # Confirms that the function creates directories when they don't exist
    def test_write_orca_input_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "new_dir")
            mol = generate_3d_molecule("CO")
            inp_path = write_orca_input(mol, nonexistent_dir, "co_test", "B3LYP def2-SVP", 0, 1)
            self.assertTrue(os.path.exists(nonexistent_dir))
            self.assertTrue(os.path.exists(inp_path))

    # Checks that empty IR spectrum blocks in ORCA output files return None for frequencies and intensities
    def test_parse_orca_output_empty_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "empty.out"
            out_file.write_text("IR SPECTRUM\n* end\n")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)

    # Ensures corrupt or invalid ORCA output files are handled gracefully
    def test_parse_orca_output_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "corrupt.out"
            out_file.write_text("nonsense garbage")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)
    
    # Verifies the function handles nonexistent files properly
    def test_parse_orca_output_file_error(self):
        freqs, intensities = parse_orca_output("nonexistent_file.out")
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)

    # Checks that auxiliary ORCA files are removed while keeping input and output files
    def test_cleanup_orca_files_removes_aux_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            keep = ["job.inp", "job.out"]
            delete = ["job.gbw", "job.xyz", "job.hess", "job.tmp"]
            for f in keep + delete:
                Path(tmpdir, f).write_text("mock")
            cleanup_orca_files(tmpdir, "job")
            for f in keep:
                self.assertTrue(Path(tmpdir, f).exists())
            for f in delete:
                self.assertFalse(Path(tmpdir, f).exists())
    
    # Ensures exceptions during file cleanup are caught and don't crash the program
    def test_cleanup_orca_files_exception_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "job.gbw").write_text("mock")

            with patch('os.remove', side_effect=Exception("Permission denied")):
                try:
                    cleanup_orca_files(tmpdir, "job")
                    self.assertTrue(True)
                except:
                    self.fail("cleanup_orca_files raised an exception!")

    # Validates that invalid ORCA paths are handled properly
    def test_run_orca_invalid_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_file = Path(tmpdir) / "fail.inp"
            inp_file.write_text("dummy")
            result = run_orca("bad_path.exe", str(inp_file), tmpdir)
            self.assertIsNone(result)
    
    # Checks that ORCA is executed correctly with valid paths and settings
    @patch('subprocess.run')
    def test_run_orca_successful_execution(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_file = Path(tmpdir) / "test.inp"
            inp_file.write_text("test input")
            result = run_orca("orca.exe", str(inp_file), tmpdir)
            self.assertIsNotNone(result)
            self.assertTrue(result.endswith(".out"))
            mock_run.assert_called_once()
    
    # Validates that a matplotlib figure is produced with proper labels from valid IR data
    def test_plot_ir_spectrum_valid_data(self):
        freqs = np.array([500.0, 1000.0, 1500.0])
        intensities = np.array([0.1, 0.5, 0.2])
        fig = plot_ir_spectrum(freqs, intensities, sigma=20, scale_factor=0.97)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Wavenumber (cm⁻¹)")
        self.assertEqual(ax.get_ylabel(), "% Transmittance")
        
    # Verifies chemical names are correctly converted to SMILES notation via PubChem
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_valid_name(self, mock_get_compounds):
        mock_compound = MagicMock()
        mock_compound.isomeric_smiles = "CCO"
        mock_get_compounds.return_value = [mock_compound]
        
        smiles = name_to_smiles("ethanol")
        self.assertEqual(smiles, "CCO")
        mock_get_compounds.assert_called_once_with("ethanol", namespace='name')
    
    # Ensures unknown chemical names return None instead of raising exceptions
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_invalid_name(self, mock_get_compounds):
        mock_get_compounds.return_value = []
        smiles = name_to_smiles("nonexistent_compound")
        self.assertIsNone(smiles)
    
    # Confirms that API exceptions from PubChem are handled gracefully
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_exception_handling(self, mock_get_compounds):
        mock_get_compounds.side_effect = Exception("API Error")
        smiles = name_to_smiles("ethanol")
        self.assertIsNone(smiles)
    
    # Validates that a 3D molecular viewer object is properly created from a molecule
    def test_mol_to_3dviewer(self):
        mol = generate_3d_molecule("CCO")
        viewer = mol_to_3dviewer(mol)
        self.assertIsNotNone(viewer)
        self.assertTrue(hasattr(viewer, '_make_html'))
    
    # Checks that the 3D molecular viewer HTML is correctly rendered
    @patch('streamlit.components.v1.html')
    def test_show_3dmol(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.return_value = "<div>3D Viewer</div>"
        show_3dmol(mock_viewer)
        mock_html.assert_called_once_with("<div>3D Viewer</div>", height=300)
    
    # Ensures exceptions in 3D viewer creation are caught and don't crash the application
    @patch('streamlit.components.v1.html')
    def test_show_3dmol_exception_handling(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.side_effect = Exception("Viewer error")
        
        try:
            show_3dmol(mock_viewer)
            self.assertTrue(True)
        except:
            self.fail("show_3dmol didn't handle the exception!")

    # Verifies psi4 frequency calculations correctly extract vibrational frequencies and IR intensities
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_successful(self, mock_frequency):
        mock_wfn = MagicMock()
        mock_freq_analysis = {
            'omega': MagicMock(data=[100.0, 200.0, 300.0]),
            'IR_intensity': MagicMock(data=[0.1, 0.5, 0.2])
        }
        mock_wfn.frequency_analysis = mock_freq_analysis
        mock_frequency.return_value = (0.0, mock_wfn)
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(intensities)
        np.testing.assert_array_almost_equal(freqs, [100.0, 200.0, 300.0])
        np.testing.assert_array_almost_equal(intensities, [0.1, 0.5, 0.2])
        self.assertTrue(ir_available)
    
    # Ensures psi4 calculation errors are handled gracefully without crashing
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_error(self, mock_frequency):
        mock_frequency.side_effect = Exception("Psi4 error")
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)
        self.assertFalse(ir_available)
    
    # Validates that missing IR intensities in psi4 output are handled properly
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_missing_ir(self, mock_frequency):
        mock_wfn = MagicMock()
        mock_freq_analysis = {
            'omega': MagicMock(data=[100.0, 200.0, 300.0]),
        }
        mock_wfn.frequency_analysis = mock_freq_analysis
        mock_frequency.return_value = (0.0, mock_wfn)
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(intensities)
        np.testing.assert_array_almost_equal(freqs, [100.0, 200.0, 300.0])
        self.assertEqual(len(intensities), 3)
        self.assertFalse(ir_available)
    
    # Checks that zero IR intensities are properly flagged as unavailable IR data
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_malformed_ir(self, mock_frequency):
        mock_wfn = MagicMock()
        mock_freq_analysis = {
            'omega': MagicMock(data=[100.0, 200.0, 300.0]),
            'IR_intensity': MagicMock(data=[0.0, 0.0, 0.0]) 
        }
        mock_wfn.frequency_analysis = mock_freq_analysis
        mock_frequency.return_value = (0.0, mock_wfn)
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNotNone(freqs)
        self.assertIsNotNone(intensities)
        self.assertEqual(len(intensities), 3)
        self.assertFalse(ir_available)
    """"
    # Tests if a valid ORCA output file is correctly parsed to extract IR frequencies and intensities
    def test_parse_orca_output_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_out = os.path.join(tmpdir, "test.out")
            with open(test_out, 'w') as f:
                f.write("""
    """""
    IR SPECTRUM
    Mode    freq    intensity
    1:      1000.0  50.0
    2:      2000.0  30.0
    """
    """"""
    """"
            freqs, inten = parse_orca_output(test_out)
            self.assertEqual(freqs, [1000.0, 2000.0])
            self.assertEqual(inten, [50.0, 30.0])
    """
    """
    # Verifies that small molecules are correctly optimized using psi4 geometry optimization
    @patch('psi4.geometry')
    def test_smiles_to_optimized_geometry_small_mol(self, mock_geo):
        mock_geo.return_value = "mock_molecule"
        mock_mol = MagicMock(spec=Chem.rdchem.Mol)
        mock_mol.GetNumAtoms.return_value = 2
        with patch('src.irs.QM_combiner.generate_3d_molecule', return_value=mock_mol):
            # Note: Fixed function call to avoid non-existent module error
            mol, rdkit_mol = smiles_to_optimized_geometry("CC", "HF/STO-3G")
            self.assertEqual(mol, "mock_molecule")
    """
    # Tests if mol_to_3dviewer correctly configures the 3D viewer with proper styling
    @patch('rdkit.Chem.MolToMolBlock')
    @patch('py3Dmol.view')
    def test_mol_to_3dviewer_configuration(self, mock_view, mock_to_molblock):
        mock_mol = MagicMock()
        mock_molblock = "MOCK MOLBLOCK"
        mock_to_molblock.return_value = mock_molblock
        mock_viewer = MagicMock()
        mock_view.return_value = mock_viewer
        
        result = mol_to_3dviewer(mock_mol)
        
        mock_to_molblock.assert_called_once_with(mock_mol)
        mock_view.assert_called_once_with(width=400, height=300)
        mock_viewer.addModel.assert_called_once_with(mock_molblock, 'mol')
        mock_viewer.setStyle.assert_called_once()
        mock_viewer.setBackgroundColor.assert_called_once_with('white')
        mock_viewer.zoomTo.assert_called_once()
        self.assertEqual(result, mock_viewer)


if __name__ == '__main__':
    unittest.main()