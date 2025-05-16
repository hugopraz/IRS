import streamlit as st
import pubchempy as pcp
import numpy as np
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import py3Dmol
import streamlit.components.v1 as components
import os
import subprocess
from pathlib import Path
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from irs.ir_Structure import build_and_plot_ir_spectrum_from_smiles

# Configuration page
st.set_page_config(page_title="IR Spectrum Simulator", layout="wide")
st.title("IR Spectrum Simulator")
st.write("Simulate IR spectra using quantum chemistry methods (Psi4 or ORCA)")

# Configuration for calculation mode sidebar
with st.sidebar:
    st.header("Configuration")
    engine = st.radio("Computational Engine", ["Psi4", "ORCA", "Functional groups"])
    if engine == "Psi4":
        method_choice = st.selectbox(
            "Computational Method:",
            ("HF/STO-3G (Fast, Rough)", "B3LYP/6-31G(d) (Balanced)", "MP2/cc-pVDZ (Slow, Accurate)")
        )
        method_mapping = {
            "HF/STO-3G (Fast, Rough)": "HF/STO-3G",
            "B3LYP/6-31G(d) (Balanced)": "B3LYP/6-31G(d)",
            "MP2/cc-pVDZ (Slow, Accurate)": "MP2/cc-pVDZ"
        }
        selected_method = method_mapping[method_choice]
    elif engine == "ORCA":
        method_choice = st.selectbox(
            "Computational Method:",
            ("B3LYP/def2-SVP (Default)", "PBE0/def2-SVP (Fast)", "wB97X-D3/def2-TZVP (Accurate)")
        )
        method_mapping = {
            "B3LYP/def2-SVP (Default)": "B3LYP def2-SVP",
            "PBE0/def2-SVP (Fast)": "PBE0 def2-SVP",
            "wB97X-D3/def2-TZVP (Accurate)": "wB97X-D3 def2-TZVP"
        }
    else:
        selected_method= "Functional Group"
    
    # ORCA path configuration
    if engine == "ORCA":
        st.subheader("ORCA Configuration")
        orca_path = st.text_input("ORCA Executable Path:", "C:/ORCA/orca.exe")
        output_dir = st.text_input("Output Directory:", "C:/temp/orca_output")

    # Width/ Frequency scaling
    st.subheader("Spectrum Settings")
    peak_width = st.slider("Peak Width (σ):", 5, 50, 20)
    freq_scale = st.slider("Frequency Scaling Factor:", 0.8, 1.1, 0.97)
    
    # Debug mode
    debug_mode = st.checkbox("Debug Mode", False)


# 1. Functions common to both QM engines
def name_to_smiles(name):
    try:
        compounds = pcp.get_compounds(name, namespace='name')
        if compounds and compounds[0].isomeric_smiles:
            return compounds[0].isomeric_smiles
    except Exception as e:
        st.warning(f"PubChem lookup failed: {e}")
    return None

# Generates 3D molecule
def generate_3d_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error(f"Could not parse SMILES: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDG()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        result = AllChem.EmbedMolecule(mol, randomSeed=42)
        if result != 0:
            st.error("3D embedding of the molecule failed.")
            return None
    try:
        AllChem.UFFOptimizeMolecule(mol, maxIters=200)
        if debug_mode:
            st.success("UFF optimization completed successfully.")
    except Exception as e:
        st.warning(f"UFF optimization did not fully converge: {e}")
    return mol

# Convert molecule to 3D viewer
def mol_to_3dviewer(mol):
    mol_block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=300)
    viewer.addModel(mol_block, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    return viewer

# Show molecule in 3D viewer
def show_3dmol(viewer):
    try:
        components.html(viewer._make_html(), height=300)
    except Exception as e:
        st.warning(f"⚠️ 3D viewer failed: {e}")

# IR-sepctra plotting function
def plot_ir_spectrum(freqs, intensities, sigma=20, scale_factor=0.97):
    """Plot IR spectrum from frequencies and intensities"""
    scaled_freqs = [f * scale_factor for f in freqs]
    x_min = max(min(scaled_freqs) - 200, 400)  
    x_max = max(scaled_freqs) + 200
    x = np.linspace(x_min, x_max, 5000)
    y = np.zeros_like(x)
    for f, inten in zip(scaled_freqs, intensities):
        if f > 0:  
            y += inten * np.exp(-((x - f) ** 2) / (2 * sigma ** 2))
    if max(y) > 0:
        y = 100 - (100 * y / max(y))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("% Transmittance")
    ax.set_title("Simulated IR Spectrum")
    ax.invert_xaxis()
    ax.set_ylim(0, 105)
    ax.grid(True)
    plt.close(fig)
    return fig

# 2. Psi4-specific functions
@st.cache_resource(show_spinner="🔄 Optimizing geometry...")

def cached_geometry_optimization(smiles, method):
    """Cache results of geometry optimization to avoid recomputation"""
    return smiles_to_optimized_geometry(smiles, method)

def smiles_to_optimized_geometry(smiles, method):
    """Convert SMILES to optimized geometry using Psi4"""
    import psi4
    mol = generate_3d_molecule(smiles)
    if mol is None:
        return None, None
    mol_block = Chem.MolToMolBlock(mol)
    mol_str = ""
    for line in mol_block.split("\n")[4:]:
        parts = line.split()
        if len(parts) >= 4:
            mol_str += f"{parts[3]} {parts[0]} {parts[1]} {parts[2]}\n"
    charge = Chem.GetFormalCharge(mol)
    unpaired_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    multiplicity = 1 + unpaired_electrons if unpaired_electrons > 0 else 1
    molecule = psi4.geometry(f"""
{charge} {multiplicity}
{mol_str}
units angstrom
""")
    try:
        if mol.GetNumAtoms() >= 3:
            st.info("⚙️ Optimizing geometry using QM method...")
            psi4.set_memory('2 GB')
            psi4.core.set_output_file('output.dat', False)
            energy, opt_wfn = psi4.optimize(method, molecule=molecule, return_wfn=True)
            st.success("✅ QM geometry optimization complete.")
        else:
            st.info("ℹ️ Skipping QM optimization — molecule too small.")
    except Exception as e:
        st.warning(f"⚠️ Optimization skipped due to error: {e}")

    return molecule, mol

def psi4_calculate_frequencies(molecule, selected_method):
    """Calculate vibrational frequencies using Psi4"""
    import psi4
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    try:
        start_time = time.time()
        energy, wfn = psi4.frequency(selected_method, molecule=molecule, return_wfn=True)
        elapsed_time = time.time() - start_time

        if debug_mode:
            available_keys = list(wfn.frequency_analysis.keys())
            st.write("📎 Available Psi4 frequency analysis keys:", available_keys)
    except Exception as e:
        st.error(f"❌ Psi4 calculation error: {e}")
        return None, None, 0.0, False
    
    freqs = np.array([float(f) for f in wfn.frequency_analysis['omega'].data])
    intensities = None
    ir_available = False
    for key in ["IR_intensity", "IR_intensities"]:
        if key in wfn.frequency_analysis:
            val = wfn.frequency_analysis[key]
            try:
                data = getattr(val, "data", val)
                intensities = np.array([float(i) for i in data])
                if np.all(intensities == 0) or np.any(np.isnan(intensities)):
                    raise ValueError("All-zero or NaN intensities")
                ir_available = True
                break
            except Exception as e:
                st.warning(f"⚠️ IR intensity data malformed: {e}")
                intensities = None
    if intensities is None:
        intensities = np.ones_like(freqs)
        st.warning("⚠️ IR intensities not found. Using dummy values.")
    else:
        st.success("✅ Real IR intensities extracted.")
        if debug_mode:
            st.write("🔬 First few IR intensities:", intensities[:5])
    return freqs, intensities, elapsed_time, ir_available

# 3. ORCA-specific functions

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
def write_orca_input(mol, output_dir, base_name, method, charge, multiplicity):
    """Generate ORCA input file for geometry optimization and IR frequency calculation"""
    os.makedirs(output_dir, exist_ok=True)
    inp_path = os.path.join(output_dir, f"{base_name}.inp")
    conf = mol.GetConformer()
    
    with open(inp_path, 'w', newline='\n') as f:
        f.write(f"! {method} Opt Freq\n")
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
def run_orca(orca_path, inp_path, output_dir):
    """Launch an ORCA job and return the output path"""
    inp_path = Path(inp_path).resolve()
    output_path = Path(output_dir) / f"{inp_path.stem}.out"
    
    try:
        with open(output_path, 'w') as out_file:
            process = subprocess.run(
                [str(orca_path), str(inp_path)],
                cwd=str(output_dir),
                stdout=out_file,
                stderr=subprocess.PIPE,
                check=True,
                text=True,
                shell=True
            )
        return str(output_path)
    except subprocess.CalledProcessError as e:
        st.error(f"❌ ORCA failed with error: {e.stderr}")
        return None
# Input: Path to .inp file
# Output: Path to .out file, or None if failed
# - Calls ORCA using subprocess with cwd set to output dir
# - Handles execution and checks for output file existence

# Extract vibrational frequencies and IR intensities from ORCA .out file
def parse_orca_output(output_path):
    """Extract vibrational frequencies and IR intensities from ORCA .out file"""
    frequencies = []
    intensities = []
    reading = False
    try:
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
            st.warning("No IR data parsed from ORCA output.")
            return None, None

        return np.array(frequencies), np.array(intensities)  
    except Exception as e:
        st.error(f"Failed to parse ORCA output: {e}")
        return None, None
# Input: ORCA output file path (.out)
# Output: List of (frequency, intensity) tuples in cm⁻¹ and km/mol
# - Looks for "IR SPECTRUM" section and reads values line by line
# - Returns None if no values found or parsing fails

# Remove temporary ORCA-generated files (except .inp and .out)
def cleanup_orca_files(output_dir, base_name):
    """Remove temporary ORCA-generated files (except .inp and .out)"""
    for filename in os.listdir(output_dir):
        if not (filename.startswith(base_name + ".") or filename.startswith(base_name + "_")):
            continue
        if filename.endswith(".inp") or filename.endswith(".out"):
            continue
        try:
            os.remove(os.path.join(output_dir, filename))
        except Exception as e:
            if debug_mode:
                st.warning(f"Could not remove file {filename}: {e}")
# Input: Base job name
# Output: None (removes matching files)
# - Keeps only .inp and .out files for debugging
# - Removes auxiliary files: .gbw, .xyz, .hess, etc.


# 4. Main layout
input_mode = st.radio("Input method", ["Molecule name", "SMILES string"])
smiles = None

col1, col2 = st.columns(2)

with col1:
    if input_mode == "Molecule name":
        molecule_name = st.text_input("Enter a molecule name (e.g., ethanol, acetone):", "ethanol")
        if molecule_name:
            smiles = name_to_smiles(molecule_name)
    else:
        smiles = st.text_input("Enter a SMILES string (e.g., CCO for ethanol):", "CCO")

with col2:
    if smiles:
        st.success(f"✅ SMILES: `{smiles}`")
        try:
            mol = generate_3d_molecule(smiles)
            if mol:
                st.image(Draw.MolToImage(mol, size=(300, 200)), caption="2D Structure")
        except Exception as e:
            st.error(f"Error generating molecule preview: {e}")

if smiles:
    st.subheader("3D Molecular Structure")
    
    try:
        mol = generate_3d_molecule(smiles)
        if mol:
            num_atoms = mol.GetNumAtoms()
            
            if num_atoms > 50:
                st.warning(f"⚠️ Molecule has {num_atoms} atoms. Calculation may be very slow or fail!")
                
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"Number of atoms: {num_atoms}")
                charge, multiplicity = guess_charge_multiplicity(mol)
                st.write(f"Charge: {charge}, Multiplicity: {multiplicity}")
                
            with col2:
                try:
                    viewer = mol_to_3dviewer(mol)
                    show_3dmol(viewer)
                except Exception as e:
                    st.error(f"Could not display 3D structure: {e}")
    except Exception as e:
        st.error(f"Error processing molecule: {e}")
    
    # Run calculations and plotting with selected engine
    with st.expander("Calculation Details", expanded=True):
        if st.button("Run IR Calculation"):
            with st.spinner(f"🔬 Running {engine} calculation for {selected_method}..."):
                if engine == "Psi4":
                    try:
                        # a) Psi4 calculation
                        import psi4
                        molecule, rdkit_mol = cached_geometry_optimization(smiles, selected_method)
                        
                        if molecule is not None:
                            freqs, intensities, elapsed_time, ir_available = psi4_calculate_frequencies(molecule, selected_method)
                            
                            if freqs is not None:
                                st.success(f"✅ Found {len(freqs)} vibrational modes in {elapsed_time:.2f} seconds")
                                valid_idx = freqs > 0
                                valid_freqs = freqs[valid_idx]
                                valid_intensities = intensities[valid_idx]
                                fig = plot_ir_spectrum(valid_freqs, valid_intensities, sigma=peak_width, scale_factor=freq_scale)
                                st.pyplot(fig)
                            else:
                                st.error("❌ Failed to calculate frequencies")
                        else:
                            st.error("❌ Failed to optimize geometry")
                    
                    except ImportError:
                        st.error("❌ Psi4 is not installed. Please install it to use this feature.")
                    
                    except Exception as e:
                        st.error(f"❌ Error during Psi4 calculation: {e}")
                
                elif engine == "ORCA": 
                    # b) ORCA calculation
                    try:
                        if not os.path.exists(orca_path):
                            st.error(f"❌ ORCA executable not found at: {orca_path}")
                        else:
                            os.makedirs(output_dir, exist_ok=True)
                            mol = generate_3d_molecule(smiles)
                            if mol is None:
                                st.error("❌ Failed to generate 3D structure")
                            else:
                                job_name = f"ir_calc_{int(time.time())}"
                                charge, multiplicity = guess_charge_multiplicity(mol)
                                st.info(f"⚙️ Preparing ORCA calculation with {selected_method}")
                                inp_path = write_orca_input(mol, output_dir, job_name, selected_method, charge, multiplicity)
                                st.info("🔬 Running ORCA calculation (this may take a while)...")
                                start_time = time.time()
                                out_path = run_orca(orca_path, inp_path, output_dir)
                                elapsed_time = time.time() - start_time
                                
                                if out_path:
                                    st.info("📊 Parsing ORCA output...")
                                    freqs, intensities = parse_orca_output(out_path)
                                    
                                    if freqs is not None and len(freqs) > 0:
                                        st.success(f"✅ Found {len(freqs)} vibrational modes in {elapsed_time:.2f} seconds")
                                        valid_idx = freqs > 0
                                        valid_freqs = freqs[valid_idx]
                                        valid_intensities = intensities[valid_idx]
                                        fig = plot_ir_spectrum(valid_freqs, valid_intensities, sigma=peak_width, scale_factor=freq_scale)
                                        st.pyplot(fig)
                                        if not debug_mode:
                                            cleanup_orca_files(output_dir, job_name)
                                    else:
                                        st.error("❌ No frequencies found in ORCA output")
                                else:
                                    st.error("❌ ORCA calculation failed")
                    
                    except Exception as e:
                        st.error(f"❌ Error during ORCA calculation: {e}")
                else:
                    try:
                        st.info("🔬 Building spectrum using functional group contributions...")
                        build_and_plot_ir_spectrum_from_smiles(smiles)
                        st.success("✅ Functional group-based IR spectrum simulated.")
                    except Exception as e:
                        st.error(f"❌ Functional group spectrum simulation failed: {e}")
st.sidebar.markdown("---")
st.sidebar.caption("Note: All calculations are performed locally using your installed quantum chemistry packages.")
st.sidebar.caption("References: Psi4 (https://psicode.org) and ORCA (https://orcaforum.kofo.mpg.de)")