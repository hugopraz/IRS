import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock
import subprocess

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
    smiles_to_optimized_geometry,
    handle_ir_calculation
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

    # Tests that generate_3d_molecule correctly creates a 3D molecule from a valid SMILES string
    def test_generate_3d_from_smiles_returns_molecule(self):
        mol = generate_3d_molecule("CCO")
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumConformers(), 0)

    # Tests that generate_3d_molecule properly adds hydrogens to the molecular structure
    def test_generate_molecule_contains_hydrogens(self):
        mol = generate_3d_molecule("C")
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        self.assertIn("H", atoms)
    
    # Tests that generate_3d_molecule handles invalid SMILES input by returning None
    def test_generate_3d_molecule_invalid_smiles(self):
        with patch('rdkit.Chem.MolFromSmiles', return_value=None):
            result = generate_3d_molecule("InvalidSMILES")
            self.assertIsNone(result)
    
    # Tests that generate_3d_molecule correctly handles failures in 3D structure embedding
    def test_generate_3d_molecule_embedding_failure(self):
        mock_mol = MagicMock()
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=1):
            result = generate_3d_molecule("C")
            self.assertIsNone(result)
    
    # Tests that generate_3d_molecule handles UFF optimization failures gracefully
    def test_generate_3d_molecule_uff_failure(self):
        mock_mol = MagicMock()
        mock_mol.GetNumConformers.return_value = 1
        
        with patch('rdkit.Chem.MolFromSmiles', return_value=mock_mol), \
             patch('rdkit.Chem.AddHs', return_value=mock_mol), \
             patch('rdkit.Chem.AllChem.EmbedMolecule', return_value=0), \
             patch('rdkit.Chem.AllChem.UFFOptimizeMolecule', side_effect=Exception("UFF failed")):
            result = generate_3d_molecule("C")
            self.assertIsNotNone(result)

    # Tests that guess_charge_multiplicity assigns charge 0 and multiplicity 1 to neutral molecules
    def test_guess_charge_and_multiplicity_neutral(self):
        mol = generate_3d_molecule("CCO")
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 0)
        self.assertEqual(multiplicity, 1)

    # Tests that guess_charge_multiplicity correctly assigns doublet multiplicity to radical species
    def test_guess_charge_and_multiplicity_radical(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
        AllChem.EmbedMolecule(mol)
        _, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(multiplicity, 2)
    
    # Tests that guess_charge_multiplicity preserves formal charge for charged molecules
    def test_guess_charge_and_multiplicity_charged(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("C[N+](C)(C)C"))
        AllChem.EmbedMolecule(mol)
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 1)
        self.assertEqual(multiplicity, 1)

    # Tests that write_orca_input creates files with correct content and structure
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
    
    # Tests that write_orca_input creates directories when they don't exist
    def test_write_orca_input_creates_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_dir = os.path.join(tmpdir, "new_dir")
            mol = generate_3d_molecule("CO")
            inp_path = write_orca_input(mol, nonexistent_dir, "co_test", "B3LYP def2-SVP", 0, 1)
            self.assertTrue(os.path.exists(nonexistent_dir))
            self.assertTrue(os.path.exists(inp_path))

    # Tests that parse_orca_output correctly extracts IR frequencies and intensities from valid output
    def test_parse_orca_output_valid(self):
        orca_output = """
                            IR SPECTRUM
    Mode      freq.(cm**-1)  T**2      TX       TY       TZ        eps.
    0:         0.00     0.00000   0.00000  0.00000  0.00000  0.00000
    1:         0.00     0.00000   0.00000  0.00000  0.00000  0.00000
    2:         0.00     0.00000   0.00000  0.00000  0.00000  0.00000
    3:      1000.00     0.50000   0.50000  0.00000  0.00000  5.50000
    4:      2000.00     0.75000   0.00000  0.75000  0.00000  7.75000
        """
        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
            f.write(orca_output)
            temp_file = f.name

        try:
            freqs, intensities = parse_orca_output(temp_file)

            self.assertIsNotNone(freqs, "Frequencies returned None")
            self.assertIsNotNone(intensities, "Intensities returned None")

            expected_freqs = np.array([0.0, 0.0, 0.0, 1000.0, 2000.0])
            expected_intensities = np.array([0.0, 0.0, 0.0, 0.5, 0.0])

            np.testing.assert_array_equal(freqs, expected_freqs)
            np.testing.assert_array_equal(intensities, expected_intensities)

        finally:
            os.unlink(temp_file)

    # Tests that parse_orca_output returns None for empty IR spectrum blocks
    def test_parse_orca_output_empty_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "empty.out"
            out_file.write_text("IR SPECTRUM\n* end\n")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)

    # Tests that parse_orca_output handles corrupt or invalid output files gracefully
    def test_parse_orca_output_fails_gracefully(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "corrupt.out"
            out_file.write_text("nonsense garbage")
            freqs, intensities = parse_orca_output(str(out_file))
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)
    
    # Tests that parse_orca_output handles nonexistent files properly
    def test_parse_orca_output_file_error(self):
        freqs, intensities = parse_orca_output("nonexistent_file.out")
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)

    # Tests that cleanup_orca_files removes auxiliary files while keeping input and output files
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
    
    # Tests that cleanup_orca_files handles exceptions during file cleanup gracefully
    def test_cleanup_orca_files_exception_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "job.gbw").write_text("mock")

            with patch('os.remove', side_effect=Exception("Permission denied")):
                try:
                    cleanup_orca_files(tmpdir, "job")
                    self.assertTrue(True)
                except:
                    self.fail("cleanup_orca_files raised an exception!")

    # Tests that run_orca handles invalid ORCA paths properly
    def test_run_orca_invalid_path_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            inp_file = Path(tmpdir) / "fail.inp"
            inp_file.write_text("dummy")
            result = run_orca("bad_path.exe", str(inp_file), tmpdir)
            self.assertIsNone(result)
    
    # Tests that run_orca executes correctly with valid paths and settings
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
    
    # Tests that plot_ir_spectrum produces a matplotlib figure with proper labels from valid IR data
    def test_plot_ir_spectrum_valid_data(self):
        freqs = np.array([500.0, 1000.0, 1500.0])
        intensities = np.array([0.1, 0.5, 0.2])
        fig = plot_ir_spectrum(freqs, intensities, sigma=20, scale_factor=0.97)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 1)
        ax = fig.axes[0]
        self.assertEqual(ax.get_xlabel(), "Wavenumber (cm⁻¹)")
        self.assertEqual(ax.get_ylabel(), "% Transmittance")
        
    # Tests that name_to_smiles correctly converts chemical names to SMILES notation via PubChem
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_valid_name(self, mock_get_compounds):
        mock_compound = MagicMock()
        mock_compound.isomeric_smiles = "CCO"
        mock_get_compounds.return_value = [mock_compound]
        
        smiles = name_to_smiles("ethanol")
        self.assertEqual(smiles, "CCO")
        mock_get_compounds.assert_called_once_with("ethanol", namespace='name')
    
    # Tests that name_to_smiles returns None for unknown chemical names
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_invalid_name(self, mock_get_compounds):
        mock_get_compounds.return_value = []
        smiles = name_to_smiles("nonexistent_compound")
        self.assertIsNone(smiles)
    
    # Tests that name_to_smiles handles PubChem API exceptions gracefully
    @patch('pubchempy.get_compounds')
    def test_name_to_smiles_exception_handling(self, mock_get_compounds):
        mock_get_compounds.side_effect = Exception("API Error")
        smiles = name_to_smiles("ethanol")
        self.assertIsNone(smiles)

    # Tests that mol_to_3dviewer correctly configures the 3D viewer with proper styling
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
    
    # Tests that mol_to_3dviewer creates a 3D molecular viewer object from a molecule
    def test_mol_to_3dviewer(self):
        mol = generate_3d_molecule("CCO")
        viewer = mol_to_3dviewer(mol)
        self.assertIsNotNone(viewer)
        self.assertTrue(hasattr(viewer, '_make_html'))

    # Tests that psi4_calculate_frequencies correctly extracts vibrational frequencies and IR intensities
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
    
    # Tests that psi4_calculate_frequencies handles calculation errors gracefully
    @patch('psi4.frequency')
    def test_psi4_calculate_frequencies_error(self, mock_frequency):
        mock_frequency.side_effect = Exception("Psi4 error")
        
        molecule = MagicMock()
        freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, "HF/STO-3G")
        
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)
        self.assertFalse(ir_available)
    
    # Tests that psi4_calculate_frequencies handles missing IR intensities in output properly
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
    
    # Tests that psi4_calculate_frequencies properly flags zero IR intensities as unavailable
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
    
    # Tests that show_3dmol correctly renders the 3D molecular viewer HTML
    @patch('streamlit.components.v1.html')
    def test_show_3dmol(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.return_value = "<div>3D Viewer</div>"
        show_3dmol(mock_viewer)
        mock_html.assert_called_once_with("<div>3D Viewer</div>", height=300)
    
    # Tests that show_3dmol handles exceptions in 3D viewer creation gracefully
    @patch('streamlit.components.v1.html')
    def test_show_3dmol_exception_handling(self, mock_html):
        mock_viewer = MagicMock()
        mock_viewer._make_html.side_effect = Exception("Viewer error")
        
        try:
            show_3dmol(mock_viewer)
            self.assertTrue(True)
        except:
            self.fail("show_3dmol didn't handle the exception!")

    # Tests that build_and_plot_ir_spectrum_from_smiles creates IR spectrum from SMILES using functional groups
    def test_build_and_plot_ir_from_valid_smiles(self):
        smiles = "CCO"
        result = build_and_plot_ir_spectrum_from_smiles(smiles)

        self.assertIsNotNone(result, "Function returned None for valid SMILES")
        self.assertIsInstance(result, tuple, "Result should be a tuple")
        self.assertEqual(len(result), 2, "Result should have two elements: x and y data")

    # Tests that build_and_plot_ir_spectrum_from_smiles handles functional group analysis errors
    @patch('src.irs.ir_Structure.analyze_molecule')
    def test_build_and_plot_ir_spectrum_from_smiles_error(self, mock_analyze):
        mock_analyze.side_effect = Exception("Invalid SMILES")
        
        with self.assertRaises(Exception):
            build_and_plot_ir_spectrum_from_smiles("invalid_smiles")

    # Tests that smiles_to_optimized_geometry correctly optimizes small molecules using psi4
    @patch('psi4.geometry')
    @patch('psi4.optimize')
    def test_smiles_to_optimized_geometry_small_mol(self, mock_optimize, mock_geo):
        mock_geo.return_value = "mock_molecule"
        mock_optimize.return_value = (0.0, MagicMock())
        mock_mol = MagicMock(spec=Chem.rdchem.Mol)
        mock_mol.GetNumAtoms.return_value = 2
        with patch('src.irs.QM_combiner.generate_3d_molecule', return_value=mock_mol), \
             patch('rdkit.Chem.MolToMolBlock', return_value="mock\n\n\n\nC 0 0 0 C\nO 1 0 0 O\n"), \
             patch('rdkit.Chem.GetFormalCharge', return_value=0):
            mol, rdkit_mol = smiles_to_optimized_geometry("CC", "HF/STO-3G")
            self.assertEqual(mol, "mock_molecule")
            self.assertEqual(rdkit_mol, mock_mol)

    # Tests that smiles_to_optimized_geometry handles large molecules without optimization
    @patch('psi4.geometry')
    def test_smiles_to_optimized_geometry_large_mol(self, mock_geo):
        mock_geo.return_value = "mock_molecule"
        mock_mol = MagicMock(spec=Chem.rdchem.Mol)
        mock_mol.GetNumAtoms.return_value = 10
        with patch('src.irs.QM_combiner.generate_3d_molecule', return_value=mock_mol), \
             patch('rdkit.Chem.MolToMolBlock', return_value="mock\n\n\n\nC 0 0 0 C\n"), \
             patch('rdkit.Chem.GetFormalCharge', return_value=0):
            mol, rdkit_mol = smiles_to_optimized_geometry("CCCCCCCCCC", "HF/STO-3G")
            self.assertEqual(mol, "mock_molecule")
            self.assertEqual(rdkit_mol, mock_mol)

    # Tests that smiles_to_optimized_geometry returns None for invalid SMILES input
    def test_smiles_to_optimized_geometry_invalid_smiles(self):
        with patch('src.irs.QM_combiner.generate_3d_molecule', return_value=None):
            mol, rdkit_mol = smiles_to_optimized_geometry("InvalidSMILES", "HF/STO-3G")
            self.assertIsNone(mol)
            self.assertIsNone(rdkit_mol)

    # Tests that name_to_smiles returns None for non-existent molecule names
    def test_pubchem_lookup_failure(self):
        smiles = name_to_smiles("notarealmolecule123")
        self.assertIsNone(smiles)

    # Tests that generate_3d_molecule fails gracefully with invalid SMILES
    def test_generate_3d_fails_with_invalid_smiles(self):
        mol = generate_3d_molecule("C(")
        self.assertIsNone(mol)

    # Tests that parse_orca_output returns None when IR SPECTRUM section is missing
    def test_parse_orca_output_no_data(self):
        with tempfile.NamedTemporaryFile('w+', suffix='.out', delete=False) as f:
            f.write("Some random content without IR SPECTRUM")
            f.flush()
        freqs, intensities = parse_orca_output(f.name)
        self.assertIsNone(freqs)
        self.assertIsNone(intensities)

    # Tests that cached_geometry_optimization skips optimization for small molecules
    @patch("src.irs.QM_combiner.generate_3d_molecule")
    def test_cached_geometry_small_molecule_skips_optimization(self, mock_gen):
        from src.irs.QM_combiner import cached_geometry_optimization
        mol = Chem.MolFromSmiles("C")
        mock_gen.return_value = mol
        molecule, _ = cached_geometry_optimization("C", "HF/STO-3G")
        self.assertIsNotNone(molecule)

    # Tests that run_orca handles subprocess failures gracefully
    @patch("src.irs.QM_combiner.subprocess.run")
    @patch("src.irs.QM_combiner.st")
    def test_run_orca_fails_gracefully(self, mock_st, mock_run):
        from src.irs.QM_combiner import run_orca
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd='orca input.inp',
            stderr="Simulated ORCA error"
        )
        
        with tempfile.NamedTemporaryFile(suffix=".inp", delete=False) as f:
            f.write(b"* xyz 0 1\nH 0.0 0.0 0.0\n*\n")
            f.flush()
            result = run_orca("orca", f.name, os.path.dirname(f.name))
            self.assertIsNone(result)
        os.unlink(f.name)

    # Tests that parse_orca_output handles incomplete or malformed IR spectrum lines
    def test_parse_orca_output_incomplete_lines(self):
        orca_output = """
                            IR SPECTRUM
        Mode      freq.(cm**-1)  T**2      TX       TY       TZ        eps.
        3:      1000.00     0.50000   BADVAL  0.00000  0.00000  5.50000
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.out', delete=False) as f:
            f.write(orca_output)
            temp_file = f.name

        try:
            freqs, intensities = parse_orca_output(temp_file)
            self.assertIsNone(freqs)
            self.assertIsNone(intensities)
        finally:
            os.unlink(temp_file)

    # Tests that handle_ir_calculation successfully processes Psi4 frequency calculations
    @patch("src.irs.QM_combiner.st")
    @patch("src.irs.QM_combiner.cached_geometry_optimization")
    @patch("src.irs.QM_combiner.psi4_calculate_frequencies")
    @patch("src.irs.QM_combiner.plot_ir_spectrum")
    def test_psi4_path_success(self, mock_plot, mock_calc, mock_opt, mock_st):
        from unittest.mock import MagicMock
        import numpy as np

        mock_opt.return_value = (MagicMock(), MagicMock())

        mock_calc.return_value = (
            np.array([1000.0, 1500.0]),
            np.array([0.5, 0.7]),
            1.23,
            True
        )

        mock_plot.return_value = MagicMock()

        handle_ir_calculation(
            smiles="CCO",
            engine="Psi4",
            selected_method="HF/STO-3G",
            orca_path="",
            output_dir="",
            freq_scale=1.0,
            peak_width=20,
            debug_mode=True
        )

        mock_plot.assert_called_once()

    # Tests that handle_ir_calculation successfully processes ORCA frequency calculations
    @patch("src.irs.QM_combiner.plot_ir_spectrum")
    @patch("src.irs.QM_combiner.parse_orca_output")
    @patch("src.irs.QM_combiner.run_orca")
    @patch("src.irs.QM_combiner.write_orca_input")
    @patch("src.irs.QM_combiner.generate_3d_molecule")
    @patch("src.irs.QM_combiner.st.text_input")
    @patch("src.irs.QM_combiner.os.path.exists")
    def test_orca_path_success(self, mock_exists, mock_st, mock_gen, mock_write, mock_run, mock_parse, mock_plot):
        mol = Chem.MolFromSmiles("CCO")
        mock_gen.return_value = mol
        mock_write.return_value = "fake_path.inp"
        mock_run.return_value = "fake_path.out"
        mock_parse.return_value = (
            [1000.0, 2000.0],  
            [1.0, 0.8]         
        )
        mock_plot.return_value = MagicMock()

        handle_ir_calculation(
            smiles="CCO",
            engine="ORCA",
            selected_method="B3LYP/def2-SVP",
            orca_path="fake_orca_path",
            output_dir="./fake_dir",
            freq_scale=0.97,
            peak_width=25,
            debug_mode=True
        )
        mock_run.assert_called_once()

    # Tests that handle_ir_calculation successfully processes functional_groups frequency calculations
    @patch("src.irs.QM_combiner.st")
    @patch("src.irs.QM_combiner.build_and_plot_ir_spectrum_from_smiles")
    def test_functional_group_success(self, mock_build, mock_st):
        handle_ir_calculation(
            smiles="CCO",
            engine="Functional groups",
            selected_method="Functional Group",
            orca_path="",
            output_dir="",
            freq_scale=1.0,
            peak_width=20,
            debug_mode=False
        )
        mock_build.assert_called_once_with("CCO")

if __name__ == '__main__':
    unittest.main()