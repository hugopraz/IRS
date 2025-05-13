import streamlit as st
import pubchempy as pcp
import psi4
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import matplotlib.pyplot as plt
import numpy as np
import py3Dmol
import streamlit.components.v1 as components

# Functions
def name_to_smiles(name):
    try:
        compounds = pcp.get_compounds(name, namespace='name')
        if compounds and compounds[0].isomeric_smiles:
            return compounds[0].isomeric_smiles
    except Exception as e:
        st.warning(f"PubChem lookup failed: {e}")
    return None

@st.cache_resource(show_spinner="🔄 Optimizing geometry...")
def cached_geometry_optimization(smiles, method):
    return smiles_to_optimized_geometry(smiles, method)

def smiles_to_optimized_geometry(smiles, method="B3LYP/6-31G(d)"):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    status = AllChem.UFFOptimizeMolecule(mol)
    if status != 0:
        st.warning("⚠️ UFF optimization did not fully converge.")

    mol_block = Chem.MolToMolBlock(mol)
    mol_str = ""
    for line in mol_block.split("\n")[4:]:
        parts = line.split()
        if len(parts) >= 4:
            mol_str += f"{parts[3]} {parts[0]} {parts[1]} {parts[2]}\n"

    molecule = psi4.geometry(f"""
0 1
{mol_str}
units angstrom
""")

    try:
        if mol.GetNumAtoms() >= 3:
            st.info("⚙️ Optimizing geometry using QM method...")
            # ⚠️ Do not use symmetry options — your Psi4 build doesn’t support it
            energy, opt_wfn = psi4.optimize(method, molecule=molecule, return_wfn=True)
            st.success("✅ QM geometry optimization complete.")
        else:
            st.info("ℹ️ Skipping QM optimization — molecule too small.")
    except Exception as e:
        st.warning(f"⚠️ Optimization skipped due to error: {e}")

    return molecule, mol

def calculate_frequencies(molecule, selected_method):
    import time
    psi4.set_memory('2 GB')
    psi4.core.set_output_file('output.dat', False)
    # psi4.set_options({"symmetry": "off", "scf_type": "df"})

    try:
        start_time = time.time()
        energy, wfn = psi4.frequency(selected_method, molecule=molecule, return_wfn=True)
        elapsed_time = time.time() - start_time

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
        st.write("🔬 First few IR intensities:", intensities[:5])

    return freqs, intensities, elapsed_time, ir_available

def plot_ir_spectrum(freqs, intensities, sigma=20):
    x = np.linspace(400, 4000, 5000)
    y = np.zeros_like(x)

    for f, inten in zip(freqs, intensities):
        y += inten * np.exp(-((x - f) ** 2) / (2 * sigma ** 2))

    y = 100 - (100 * y / max(y))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel("% Transmittance")
    ax.set_title("Simulated IR Spectrum")
    ax.invert_xaxis()
    ax.set_ylim(0, 105)
    ax.grid(True)
    plt.close(fig)
    return fig

def mol_to_3dviewer(mol):
    mol_block = Chem.MolToMolBlock(mol)
    viewer = py3Dmol.view(width=400, height=300)
    viewer.addModel(mol_block, 'mol')
    viewer.setStyle({'stick': {}})
    viewer.setBackgroundColor('white')
    viewer.zoomTo()
    return viewer

def show_3dmol(viewer):
    try:
        components.html(viewer._make_html(), height=300)
    except Exception as e:
        st.warning(f"⚠️ 3D viewer failed: {e}")

# Streamlit App

st.set_page_config(page_title="IR Spectrum Simulator", layout="centered")
st.title("IR Spectrum Simulator (QM-Based with Psi4)")
st.write("This app simulates the IR spectrum of a molecule based on its name or SMILES.")

method_choice = st.selectbox(
    "Choose the computational method:",
    ("HF/STO-3G (Fast, Rough)", "B3LYP/6-31G(d) (Balanced)", "MP2/cc-pVDZ (Slow, Accurate)")
)

method_mapping = {
    "HF/STO-3G (Fast, Rough)": "HF/STO-3G",
    "B3LYP/6-31G(d) (Balanced)": "B3LYP/6-31G(d)",
    "MP2/cc-pVDZ (Slow, Accurate)": "MP2/cc-pVDZ"
}

selected_method = method_mapping[method_choice]

input_mode = st.radio("Input method", ["Molecule name", "SMILES string"])
smiles = None
molecule_name = None

if input_mode == "Molecule name":
    molecule_name = st.text_input("Enter a molecule name (e.g., ethanol, acetone):", "ethanol")
    if molecule_name:
        smiles = name_to_smiles(molecule_name)
else:
    smiles = st.text_input("Enter a SMILES string (e.g., CCO for ethanol):", "CCO")

if smiles:
    st.success(f"✅ SMILES found: `{smiles}`")

    st.subheader("Step 1: Molecule Building and Optimization")
    try:
        molecule, rdkit_mol = cached_geometry_optimization(smiles, selected_method)
        num_atoms = rdkit_mol.GetNumAtoms()

        st.image(Draw.MolToImage(rdkit_mol, size=(300, 300)), caption=f"2D Structure ({num_atoms} atoms)")

        if num_atoms > 20:
            st.warning(f"⚠️ Molecule has {num_atoms} atoms. Calculation may be slow or risky!")

        viewer = mol_to_3dviewer(rdkit_mol)
        st.subheader("🔄 3D Viewer")
        show_3dmol(viewer)

        st.subheader("Step 2: Frequency Calculation (Psi4)")
        with st.spinner("🔬 Running quantum mechanical calculation... please wait!"):
            freqs, intensities, elapsed_time, ir_available = calculate_frequencies(molecule, selected_method)

        if freqs is None:
            st.error("❌ Psi4 calculation failed. Try another molecule or method.")
            st.stop()

        st.success(f"✅ Found {len(freqs)} vibrational modes.")
        st.info(f"⏱️ Calculation time: {elapsed_time:.2f} seconds.")

        if not ir_available:
            st.warning("⚠️ No IR intensities found. Using uniform dummy values for plotting.")

        fig = plot_ir_spectrum(freqs, intensities)
        st.pyplot(fig)

        st.caption("💡 Tip: OH stretch appears around 3200–3600 cm⁻¹, C=O around 1650–1750 cm⁻¹, C–H around 2800–3100 cm⁻¹.")

    except Exception as e:
        st.error(f"❌ Error occurred: {e}")
else:
    st.error("❌ Could not find SMILES or it is invalid.")
