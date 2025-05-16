import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
from irs.ir_ORCA import (
    generate_3d_from_smiles, guess_charge_multiplicity, write_orca_input,
    estimate_peak_width, estimate_gaussian_sigma, parse_orca_output,
    cleanup_files, run_orca_from_smiles, run_orca, gaussian
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

    # Confirms peak width increases with frequency
    def test_estimate_peak_width_increases_with_freq(self):
        self.assertGreater(estimate_peak_width(3000), estimate_peak_width(500))

    # Ensures sigma is positive and logically bounded
    def test_estimate_gaussian_sigma_logically_bound(self):
        freq = 1000
        sigma = estimate_gaussian_sigma(freq)
        self.assertGreater(sigma, 0)
        self.assertLess(sigma, estimate_peak_width(freq))

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

    # Ensures invalid SMILES is rejected gracefully
    def test_run_orca_from_smiles_invalid_smiles(self):
        from irs import ir_ORCA
        original_func = ir_ORCA.generate_3d_from_smiles
        ir_ORCA.generate_3d_from_smiles = lambda x: None
        try:
            self.assertIsNone(run_orca_from_smiles("??"))
        finally:
            ir_ORCA.generate_3d_from_smiles = original_func

    # Validates Gaussian peak location and maximum value
    def test_gaussian_peak_maximum(self):
        x = np.linspace(-5, 5, 1000)
        mu = 0
        sigma = 1
        y = gaussian(x, mu, sigma)
        peak_index = np.argmax(y)
        self.assertAlmostEqual(x[peak_index], mu, delta=1e-2)
        self.assertAlmostEqual(y[peak_index], 1.0, places=5)

    # Confirms symmetry of Gaussian about its mean
    def test_gaussian_symmetry_about_mu(self):
        x = np.linspace(-10, 10, 10001)
        mu = 2.0
        sigma = 1.5
        y = gaussian(x, mu, sigma)
        left = y[:len(y)//2]
        right = y[len(y)//2+1:][::-1]
        np.testing.assert_allclose(left, right, rtol=1e-5)

    # Checks Gaussian full width at half maximum ≈ 2.355 * sigma
    def test_gaussian_width_property(self):
        mu = 0
        sigma = 1
        x = np.linspace(-10, 10, 10001)
        y = gaussian(x, mu, sigma)
        half_max = 0.5
        indices = np.where(y >= half_max)[0]
        width = x[indices[-1]] - x[indices[0]]
        self.assertAlmostEqual(width, 2.355 * sigma, delta=0.05)

    # Confirms peak width increases with frequency across range
    def test_estimate_peak_width_monotonic(self):
        freqs = np.linspace(0, 4000, 100)
        widths = [estimate_peak_width(f) for f in freqs]
        self.assertTrue(all(w2 > w1 for w1, w2 in zip(widths, widths[1:])))

    # Verifies FWHM to sigma conversion is consistent
    def test_estimate_gaussian_sigma_consistency(self):
        freqs = [400, 800, 1600, 3200]
        for f in freqs:
            fwhm = estimate_peak_width(f)
            sigma = estimate_gaussian_sigma(f)
            expected_sigma = 0.425 * fwhm
            self.assertAlmostEqual(sigma, expected_sigma, places=5)

    # Tests that area under Gaussian scales linearly with sigma
    def test_gaussian_area_scaling(self):
        x = np.linspace(-50, 50, 10000)
        sigma_values = [0.5, 1, 2]
        areas = [np.trapz(gaussian(x, 0, s), x) for s in sigma_values]
        ratios = [areas[i+1] / areas[i] for i in range(len(areas)-1)]
        expected_ratios = [2, 2]
        for r, er in zip(ratios, expected_ratios):
            self.assertAlmostEqual(r, er, delta=0.2)

if __name__ == '__main__':
    unittest.main()
