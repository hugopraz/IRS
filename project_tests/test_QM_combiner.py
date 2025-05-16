import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
import io
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    show_3dmol
)

class TestOrcaFunctions(unittest.TestCase):
    # Generates a 3D molecule with hydrogens from a valid SMILES
    def test_generate_3d_from_smiles_returns_molecule(self):
        mol = generate_3d_molecule("CCO")
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumConformers(), 0)

    # Checks that hydrogens are added to the structure
    def test_generate_molecule_contains_hydrogens(self):
        mol = generate_3d_molecule("C")
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        self.assertIn("H", atoms)
    
    # Tests if the function handles invalid SMILES correctly
    def test_generate_3d_molecule_invalid_smiles(self):
        with patch('rdkit.Chem.MolFromSmiles', return_value=None):
            result = generate_3d_molecule("InvalidSMILES")
            self.assertIsNone(result)
    
    # Tests if the function handles embedding failures correctly
    def test_generate_3d_molecule_embedding_failure(self):
        mock_mol = MagicMock()
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=1), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=1):
            result = generate_3d_molecule("C")
            self.assertIsNone(result)
    
    # Tests if UFF optimization failures are handled gracefully
    def test_generate_3d_molecule_uff_failure(self):
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1
        
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=0), \
             patch('rdkit.Chem.AllChem.UFFOptimizeMolecule', side_effect=Exception("UFF failed")):
            result = generate_3d_molecule("C")
            self.assertIsNotNone(result)

    # Verifies correct charge and singlet multiplicity
    def test_guess_charge_and_multiplicity_neutral(self):
        mol = generate_3d_molecule("CCO")
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 0)
        self.assertEqual(multiplicity, 1)

    # Verifies multiplicity for radical species
    def test_guess_charge_and_multiplicity_radical(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
        AllChem.EmbedMolecule(mol)
        _, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(multiplicity, 2)
    
    # Tests if the function correctly handles charged molecules
    def test_guess_charge_and_multiplicity_charged(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[N+](C)(C)C"))
        AllChem.EmbedMolecule(mol)
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 1)
        self.assertEqual(multiplicity, 1)

    # Confirms ORCA .inp file is created and contains atoms
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
    
    # Tests if the function creates directories when they don't exist
    def test_write_orca_input_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "new_dir")
            mol = generate_3d_molecule("CO")
            inp_path = write_orca_input(mol, nonexistent_dir, "co_test", "B3LYP def2-SVP", 0, 1)
            self.assertTrue(os.path.exists(nonexistent_dir))
            self.assertTrue(os.path.exists(inp_path))

    # Verifies function returns None on empty IR spectrum block
    def test_parse_orca_output_empty_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "empty.out"
            out_file.write_text("IR SPECTRUM\n* end\n")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)

    # Confirms safe handling of corrupt or incomplete ORCA output
    def test_parse_orca_output_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "corrupt.out"
            out_file.write_text("nonsense garbage")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)
    
    # Tests if function handles file open exceptions
    def test_parse_orca_output_file_error(self):
        freqs, intensities = parse_orca_output("nonexistent_file.out")
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)

    # Removes all generated ORCA files except .inp and .out
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
    
    # Tests if function properly handles file removal errors
    def test_cleanup_orca_files_exception_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "job.gbw").write_text("mock")

            with patch('os.remove', side_effect=Exception("Permission denied")):
                try:
                    cleanup_orca_files(tmpdir, "job")
                    self.assertTrue(True)
                except:
                    self.fail("cleanup_orca_files raised an exception!")

    def test_run_orca_invalid_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_file = Path(tmpdir) / "fail.inp"
            inp_file.write_text("dummy")
            result = run_orca("bad_path.exe", str(inp_file), tmpdir)
            self.assertIsNone(result)
    
    # Tests if function correctly executes ORCA with valid paths
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
    
    # Tests if plot_ir_spectrum correctly generates a figure with valid data
    def test_plot_ir_spectrum_valid_data(self):
        freqs = np.array([500.0, 1000.0, 1500.0])
        intensities = np.array([0.1, 0.5, 0.2])
        fig = plot_ir_spectrum(freqs, intensities, sigma=20, scale_factor=0.97)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Wavenumber (cm⁻¹)")
        self.assertEqual(ax.get_ylabel(), "% Transmittance")
        
    # Tests if name_to_smiles converts common molecule names to SMILES correctly
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_valid_name(self, mock_get_compounds):
        mock_compound = MagicMock()
        mock_compound.isomeric_smiles = "CCO"
        mock_get_compounds.return_value = [mock_compound]
        
        smiles = name_to_smiles("ethanol")
        self.assertEqual(smiles, "CCO")
        mock_get_compounds.assert_called_once_with("ethanol", namespace='name')
    
    # Tests if name_to_smiles returns None for invalid molecule names
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_invalid_name(self, mock_get_compounds):
        mock_get_compounds.return_value = []
        smiles = name_to_smiles("nonexistent_compound")
        self.assertIsNone(smiles)
    
    # Tests if name_to_smiles handles exceptions from PubChem API
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_exception_handling(self, mock_get_compounds):
        mock_get_compounds.side_effect = Exception("API Error")
        smiles = name_to_smiles("ethanol")
        self.assertIsNone(smiles)
    
    # Tests if mol_to_3dviewer creates a valid 3D viewer
    def test_mol_to_3dviewer(self):
        mol = generate_3d_molecule("CCO")
        viewer = mol_to_3dviewer(mol)
        self.assertIsNotNone(viewer)
        self.assertTrue(hasattr(viewer, '_make_html'))
    
    # Tests if show_3dmol correctly renders HTML components
    @patch('streamlit.components.v1.html')
    def test_show_3dmol(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.return_value = "<div>3D Viewer</div>"
        show_3dmol(mock_viewer)
        mock_html.assert_called_once_with("<div>3D Viewer</div>", height=300)
    
    # Tests if show_3dmol handles exceptions gracefully
    @patch('streamlit.components.v1.html')
    def test_show_3dmol_exception_handling(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.side_effect = Exception("Viewer error")
        
        try:
            show_3dmol(mock_viewer)
            self.assertTrue(True)
        except:
            self.fail("show_3dmol didn't handle the exception!")

    # Tests if psi4_calculate_frequencies correctly extracts frequencies and intensities
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
    
    # Tests if psi4_calculate_frequencies handles calculation errors
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_error(self, mock_frequency):
        mock_frequency.side_effect = Exception("Psi4 error")
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)
        self.assertFalse(ir_available)
    
    # Tests if psi4_calculate_frequencies handles missing IR intensities
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
    
    # Tests if psi4_calculate_frequencies handles malformed IR data
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


if __name__ == '__main__':
    unittest.main()