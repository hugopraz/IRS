import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.irs.QM_combiner import (    
)

class TestOrcaFunctions(unittest.TestCase):

    # Tests that a valid 3D-optimized molecule is generated from SMILES
    def test_generate_3d_from_smiles_returns_molecule(self):
        mol = generate_3d_from_smiles("CCO")
        self.assertIsNotNone(mol)
        self.assertGreater(mol.GetNumConformers(), 0)

    # Confirms hydrogens are added to the generated molecule
    def test_generate_molecule_hydrogens(self):
        mol = generate_3d_from_smiles("C")
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        self.assertIn("H", symbols)

    # Verifies correct formal charge and multiplicity for neutral molecules
    def test_guess_charge_and_multiplicity(self):
        mol = generate_3d_from_smiles("CCO")
        charge, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(charge, 0)
        self.assertEqual(multiplicity, 1)

    # Verifies radical species are correctly assigned multiplicity 2
    def test_guess_radical_multiplicity(self):
        mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
        AllChem.EmbedMolecule(mol)
        _, multiplicity = guess_charge_multiplicity(mol)
        self.assertEqual(multiplicity, 2)

    # Checks that the ORCA input file contains atom symbols and coordinates
    def test_write_orca_input_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mol = generate_3d_from_smiles("CO")
            path = write_orca_input(mol, os.path.join(tmpdir, "co_test"), 0, 1)
            self.assertTrue(os.path.exists(path))
            content = open(path).read()
            self.assertIn("C", content)
            self.assertIn("O", content)
    # Parses mocked ORCA output and verifies extracted values
    def test_parse_orca_output_valid_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "mock_output.out"
            file.write_text("IR SPECTRUM\n1: 500.0 cm**-1 1.0 km/mol\n2: 1000.0 cm**-1 2.0 km/mol\n* end\n")
            result = parse_orca_output(str(file))
            self.assertEqual(result, [(500.0, 1.0), (1000.0, 2.0)])

    # Ensures parser ignores noise and still extracts valid spectrum data
    def test_parse_orca_output_handles_noise(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "noisy_output.out"
            file.write_text("IR SPECTRUM\n1: 700.0 cm**-1 3.0 km/mol\nhello world\n2: 1200.0 cm**-1 2.5 km/mol\n* end")
            result = parse_orca_output(str(file))
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0][1], 3.0)

    # Checks graceful handling of malformed spectrum block
    def test_parse_orca_output_empty_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file = Path(tmpdir) / "bad.out"
            file.write_text("IR SPECTRUM\n* end")
            self.assertIsNone(parse_orca_output(str(file)))

    # Ensures cleanup function deletes only auxiliary files, not .inp/.out
    def test_cleanup_files_deletes_only_extras(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            keep = ["test.inp", "test.out"]
            remove = ["test.tmp", "test_aux.chk"]
            for f in keep + remove:
                (run_dir / f).write_text("data")
            global OUTPUT_BASE_DIR
            OUTPUT_BASE_DIR = run_dir
            cleanup_files("test")
            for f in keep:
                self.assertTrue((run_dir / f).exists())
            for f in remove:
                self.assertFalse((run_dir / f).exists())

    # Verifies ORCA runner handles subprocess failure gracefully
    def test_run_orca_invalid_path(self):
        import subprocess
        original_run = subprocess.run
        subprocess.run = lambda *a, **k: (_ for _ in ()).throw(Exception("boom"))
        try:
            self.assertIsNone(run_orca("nonexistent.inp"))
        finally:
            subprocess.run = original_run
if __name__ == '__main__':
    unittest.main()
