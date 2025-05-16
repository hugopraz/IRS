import sys
import subprocess
from rdkit import Chem
from rdkit.Chem import AllChem
import matplotlib.pyplot as plt
import os 
import numpy as np
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from absolute path
env_path = Path(__file__).parent.parent.parent / "setup" / "orca.env"
load_dotenv(dotenv_path=env_path, override=True)

# Convert paths to raw Windows format and resolve them
ORCA_PATH = Path(os.getenv("ORCA_PATH").strip('"').replace('/', '\\')).resolve()
OUTPUT_BASE_DIR = Path(os.getenv("OUTPUT_BASE_DIR").strip('"').replace('/', '\\')).resolve()

# Verify paths exist
if not ORCA_PATH.exists():
    raise FileNotFoundError(f"❌ ORCA executable not found at: {ORCA_PATH}")
OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)

# Convert a SMILES string into a 3D-optimized RDKit molecule (pre-optimisation)
def generate_3d_from_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"ERROR: Could not parse SMILES: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            print("ERROR: 3D embedding of the molecule failed.")
            return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
    except Exception:
        print("WARNING: UFF optimization did not fully converge.")
    return mol
# Input: SMILES string (e.g. "CCO")
# Output: RDKit Mol object with 3D geometry optimized by UFF
# - Adds hydrogens, generates conformer, optimizes with force field.
# - Returns None if parsing or 3D generation fails.


# Estimate the formal charge and multiplicity of a molecule
def guess_charge_multiplicity(mol):
    charge = Chem.GetFormalCharge(mol)
    unpaired_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    multiplicity = 1 + unpaired_electrons if unpaired_electrons > 0 else 1
    return charge, multiplicity
# Input: RDKit Mol
# Output: (charge: int, multiplicity: int)
# - Charge from RDKit's GetFormalCharge
# - Multiplicity = 1 + number of unpaired electrons (if any)


# Generate ORCA input file for geometry optimization and IR frequency calculation
def write_orca_input(mol, base_name: str, charge: int, multiplicity: int):
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    inp_path = os.path.join(OUTPUT_BASE_DIR, base_name + ".inp")
    conf = mol.GetConformer()
    with open(inp_path, 'w', newline='\n') as f:
        f.write("! B3LYP def2-SVP Opt Freq\n")
        f.write(f"* xyz {charge} {multiplicity}\n")
        for atom in mol.GetAtoms():
            pos = conf.GetAtomPosition(atom.GetIdx())
            f.write(f"{atom.GetSymbol()} {pos.x:.6f} {pos.y:.6f} {pos.z:.6f}\n")
        f.write("*\n")
    return inp_path
# Input: RDKit Mol, job base name, charge, multiplicity
# Output: Path to .inp file
# - Uses B3LYP/def2-SVP with frequency calculation and optimization
# - Writes atomic coordinates in XYZ format


# Launch an ORCA job and matches the output to a file
def run_orca(inp_path: str):
    inp_path = Path(inp_path).resolve()
    output_path = OUTPUT_BASE_DIR / f"{inp_path.stem}.out"
    try:
        with open(output_path, 'w') as out_file:
            subprocess.run(
                [str(ORCA_PATH), str(inp_path)],
                cwd=str(OUTPUT_BASE_DIR),
                stdout=out_file,
                stderr=subprocess.PIPE,
                check=True,
                shell=True,  
                text=True
            )
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"❌ ORCA failed with error: {e.stderr}")
        return None
# Input: Path to .inp file
# Output: Path to .out file, or None if failed
# - Calls ORCA using subprocess with cwd set to output dir
# - Handles execution and checks for output file existence


# Extract vibrational frequencies and IR intensities from ORCA .out file
def parse_orca_output(output_path: str):
    frequencies = []
    intensities = []
    reading = False

    with open(output_path, 'r') as f:
        for line in f:
            if "IR SPECTRUM" in line:
                reading = True
                continue
            if reading:
                if "Mode" in line or "cm**-1" in line or line.strip() == "":
                    continue
                if line.strip().startswith("*") or "eps" in line:
                    break
                parts = line.strip().split()
                if len(parts) >= 4 and parts[0].endswith(":"):
                    try:
                        freq = float(parts[1])
                        inten = float(parts[3])
                        frequencies.append(freq)
                        intensities.append(inten)
                    except ValueError:
                        continue

    if not frequencies or not intensities:
        print("No IR data parsed.")
        return None

    return list(zip(frequencies, intensities))
# Input: ORCA output file path (.out)
# Output: List of (frequency, intensity) tuples in cm⁻¹ and km/mol
# - Looks for "IR SPECTRUM" section and reads values line by line
# - Returns None if no values found or parsing fails



# Remove temporary ORCA-generated files (except .inp and .out)
def cleanup_files(base_name: str):
    for filename in os.listdir(OUTPUT_BASE_DIR):
        if not (filename.startswith(base_name + ".") or filename.startswith(base_name + "_")):
            continue
        if filename.endswith(".inp") or filename.endswith(".out"):
            continue
        try:
            os.remove(os.path.join(OUTPUT_BASE_DIR, filename))
        except Exception as e:
            print(f"WARNING: Could not remove file {filename}: {e}")
# Input: Base job name
# Output: None (removes matching files)
# - Keeps only .inp and .out files for debugging
# - Removes auxiliary files: .gbw, .xyz, .hess, etc.


# Full pipeline: from SMILES to IR peaks using ORCA
def run_orca_from_smiles(smiles: str, name: str = None):
    print(f"Starting IR analysis for SMILES: {smiles}")
    mol = generate_3d_from_smiles(smiles)
    if mol is None:
        return None
    print("Generated 3D coordinates (UFF optimization complete).")
    base_name = name if name is not None else "ORCA_job"
    charge, multiplicity = guess_charge_multiplicity(mol)
    print(f"Using charge = {charge}, multiplicity = {multiplicity}")
    inp_path = write_orca_input(mol, base_name, charge, multiplicity)
    print(f"ORCA input file created: {inp_path}")
    out_path = run_orca(inp_path)
    if out_path is None:
        return None
    print(f"ORCA run finished. Output file: {out_path}")
    results = parse_orca_output(out_path)
    cleanup_files(base_name)
    if results:
        print("Vibrational wavenumbers and IR intensities:")
        for freq, inten in results:
            print(f"  {freq:.2f} cm^-1, IR intensity: {inten:.2f} km/mol")
    else:
        print("No vibrational frequency data found in output.")
    return results
# Input: SMILES string and optional job name
# Output: List of (frequency, intensity) tuples or None if error
# Steps:
# 1. Generate 3D geometry
# 2. Write ORCA input file
# 3. Run ORCA calculation
# 4. Parse resulting .out file
# 5. Clean up auxiliary files
# 6. Return vibrational mode list


def gaussian(x, mu, sigma):
    return np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


# Estimate peak FWHM (full width at half max) based on frequency
def estimate_peak_width(freq):
    return 5 + 0.01 * freq
# Input: Frequency in cm⁻¹
# Output: Estimated width (FWHM) in cm⁻¹
# - Empirical formula that broadens peaks slightly at higher frequencies


# Convert FWHM to Gaussian σ (standard deviation)
def estimate_gaussian_sigma(freq):
    return 0.425 * estimate_peak_width(freq)  
# Input: Frequency in cm⁻¹
# Output: Sigma value for Gaussian convolution
# - Uses conversion factor from FWHM to σ: σ ≈ FWHM / 2.355 ≈ 0.425 × FWHM


def plot_ir_spectrum(wavenumbers, intensities, widths=None, resolution=2, scale_freq=0.97):
    scaled_wavenumbers = [f * scale_freq for f in wavenumbers]
    scaled_intensities = [i**0.6 for i in intensities]
    max_intensity = max(scaled_intensities)
    rel_intensities = [i / max_intensity for i in scaled_intensities]

    wn_min = min(scaled_wavenumbers) - 100
    wn_max = max(scaled_wavenumbers) + 100
    wn_grid = np.arange(wn_min, wn_max, resolution)
    spectrum = np.zeros_like(wn_grid)

    for i, (wn, inten) in enumerate(zip(scaled_wavenumbers, rel_intensities)):
        sigma = widths[i] if widths else estimate_gaussian_sigma(wn)
        mask = np.abs(wn_grid - wn) <= 3 * sigma
        spectrum[mask] += inten * gaussian(wn_grid[mask], wn, sigma)

    plt.figure(figsize=(10, 5))
    plt.plot(wn_grid, -spectrum, color='black', lw=1.5)

    for wn, inten in zip(scaled_wavenumbers, rel_intensities):
        plt.axvline(x=wn, ymin=0, ymax=inten, color='gray', alpha=0.3, linewidth=0.5)

    plt.title("Simulated IR Spectrum (ORCA-based)")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Relative Absorbance (a.u.)")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
#1: All intensities are scaled relative to the strongest peak to ensure consistent plotting (range: 0 to 1)
#2: Creates a uniform grid of wavenumbers to evaluate the spectrum over a broader range than just the input peaks (adds margin)
#3: Creates an empty spectrum array (all zeros) with the same shape as wn_grid
#4: Iterates over each wavenumber and its normalized intensity, dynamic_width: peak width depends on the position (wavenumber)-> higher wavenumber = broader peak
#   mask: only compute Gaussian values where they are relevant (within ±3σ), which speeds up computation
#   +=: superimpose all broadened peaks into the final spectrum
#5: plots


# Command-line interface for running IR spectrum generation
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ir_ORCA.py \"SMILES_STRING\" [JobName]")
        sys.exit(1)

    smiles_input = sys.argv[1]
    job_name = sys.argv[2] if len(sys.argv) > 2 else "molecule"

    results = run_orca_from_smiles(smiles_input, job_name)
    if results is None:
        print("ORCA failed or parsing failed.")
        sys.exit(1)
    w, i = zip(*results)
    widths = [estimate_gaussian_sigma(f) for f in w]
    if w and i:
        plot_ir_spectrum(w, i, widths=widths)
# Expects: python ir_ORCA.py "SMILES" [Optional_JobName]
# Example: python ir_ORCA.py "CCO" ethanol
# Plots the resulting IR spectrum using matplotlib

