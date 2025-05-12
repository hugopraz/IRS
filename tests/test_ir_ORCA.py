import os
import tempfile
from rdkit import Chem
from rdkit.Chem import AllChem
from IRS.src.irs.ir_ORCA import (
    generate_3d_from_smiles, guess_charge_multiplicity, write_orca_input,
    estimate_peak_width, estimate_gaussian_sigma, parse_orca_output,
    cleanup_files, run_orca_from_smiles, run_orca
)


# Temporary directory for file-based tests
TEST_BASE_DIR = tempfile.mkdtemp()

# Checks if a valid molecule is generated and has 3D coordinates
def test_generate_3d_from_smiles_returns_molecule():
    mol = generate_3d_from_smiles("CCO")
    assert mol is not None
    assert mol.GetNumConformers() > 0

# Confirms hydrogens were added to the molecule
def test_generate_molecule_hydrogens():
    mol = generate_3d_from_smiles("C")
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    assert "H" in symbols

# Checks formal charge and spin multiplicity inference
def test_guess_charge_and_multiplicity():
    mol = generate_3d_from_smiles("CCO")
    charge, multiplicity = guess_charge_multiplicity(mol)
    assert charge == 0
    assert multiplicity == 1

# Verifies unpaired electron detection in radicals
def test_guess_radical_multiplicity():
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    AllChem.EmbedMolecule(mol)
    multiplicity = guess_charge_multiplicity(mol)
    assert multiplicity == 2

# Tests that input file contains atoms and coordinates
def test_write_orca_input_file(tmp_path):
    mol = generate_3d_from_smiles("CO")
    path = write_orca_input(mol, tmp_path / "co_test", 0, 1)
    assert os.path.exists(path)
    content = open(path).read()
    assert "C" in content and "O" in content
    os.remove(path)

# Checks peak width scales with frequency
def test_estimate_peak_width_increases_with_freq():
    assert estimate_peak_width(3000) > estimate_peak_width(500)

# Sigma should be positive and proportional to width
def test_estimate_gaussian_sigma_logically_bound():
    freq = 1000
    sigma = estimate_gaussian_sigma(freq)
    assert 0 < sigma < estimate_peak_width(freq)

# Mocks a valid ORCA output and tests parsed frequencies/intensities
def test_parse_orca_output_valid_data(tmp_path):
    file = tmp_path / "mock_output.out"
    file.write_text("IR SPECTRUM\n1: 500.0 cm**-1 1.0 km/mol\n2: 1000.0 cm**-1 2.0 km/mol\n* end\n")
    result = parse_orca_output(str(file))
    assert result == [(500.0, 1.0), (1000.0, 2.0)]

# Handles malformed ORCA output with extra lines
def test_parse_orca_output_handles_noise(tmp_path):
    file = tmp_path / "noisy_output.out"
    file.write_text("IR SPECTRUM\n1: 700.0 cm**-1 3.0 km/mol\nhello world\n2: 1200.0 cm**-1 2.5 km/mol\n* end")
    result = parse_orca_output(str(file))
    assert len(result) == 2
    assert result[0][1] == 3.0

# Fails gracefully if no valid data parsed
def test_parse_orca_output_empty_block(tmp_path):
    file = tmp_path / "bad.out"
    file.write_text("IR SPECTRUM\n* end")
    assert parse_orca_output(str(file)) is None

# Tests cleanup function only deletes non .inp/.out files
def test_cleanup_files_deletes_only_extras(tmp_path):
    run_dir = tmp_path / "ORCA_runs"
    os.makedirs(run_dir, exist_ok=True)
    base = "test"
    keep = [f"{base}.inp", f"{base}.out"]
    remove = [f"{base}.tmp", f"{base}_aux.chk"]
    for fname in keep + remove:
        (run_dir / fname).write_text("data")
    global OUTPUT_BASE_DIR
    OUTPUT_BASE_DIR = str(run_dir)
    cleanup_files(base)
    for fname in keep:
        assert os.path.exists(run_dir / fname)
    for fname in remove:
        assert not os.path.exists(run_dir / fname)

# Fails gracefully if ORCA path is invalid
def test_run_orca_invalid_path(monkeypatch):
    monkeypatch.setattr("subprocess.run", lambda *a, **k: (_ for _ in ()).throw(Exception("boom")))
    assert run_orca("nonexistent.inp") is None

# Returns None for invalid SMILES input
def test_run_orca_from_smiles_invalid_smiles(monkeypatch):
    monkeypatch.setattr("src.irs.ir_ORCA.generate_3d_from_smiles", lambda x: None)
    assert run_orca_from_smiles("??") is None
