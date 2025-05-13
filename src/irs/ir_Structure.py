from rdkit import Chem
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter


FUNCTIONAL_GROUPS = {
    "Fluoroalkane": "[#6][F]",
    "Chloroalkane": "[#6][Cl]",
    "Bromoalkane": "[#6][Br]",
    "Iodoalkane": "[#6][I]",
    "Alcohol": "[OX2H][#6]",
    "Phenol": "[OX2H][a]",
    "Ether": "[OD2]([#6])[#6]",
    "Epoxide": "[#6]1[#6][OX2]1",
    "Aldehyde": "[CX3H1](=O)[#6]",
    "Ketone": "[#6][CX3](=O)[#6]",
    "Carboxylic Acid": "[CX3](=O)[OX2H1]",
    "Ester": "[#6][CX3](=O)[OX2H0][#6]",
    "Lactone": "[CX3](=O)[OX2][#6]@[#6]",
    "Acid Anhydride": "[#6][CX3](=O)[OX2][CX3](=O)[#6]",
    "Acyl Halide": "[CX3](=O)[F,Cl,Br,I]",
    "Peroxide": "[#6][OX2][OX2][#6]",
    "Hydroperoxide": "[#6][OX2][OX2H]",
    "Peracid": "[CX3](=O)[OX2][OX2H]",
    "Ketone": "[#6]=[#8]",
    "Amine (Primary)": "[NX3H2][#6]",
    "Amine (Secondary)": "[NX3H1]([#6])[#6]",
    "Amine (Tertiary)": "[NX3H0]([#6])([#6])[#6]",
    "Amide": "[NX3][CX3](=O)[#6]",
    "Lactam": "[#6]1[#6][#6][CX3](=O)[NX3]1",
    "Imine": "[NX2]=[CX3]",
    "Nitrile": "[NX1]#[CX2]",
    "Isocyanate": "[NX2]=[CX2]=[OX1]",
    "Hydrazine": "[N]-[N]",
    "Azide": "N=[N+]=[N-]",
    "Diazenyl": "[N]=[N]",
    "Guanidine": "[NX3H1,NX3H2][CX3](=[NX2])[NX3H1,NX3H2]",
    "N-Oxide (charged)": "[#6][N+]([O-])[#6]", 
    "N-Oxide (acyclic)": "[N+]([O-])([#6])[#6]", 
    "N-Oxide (nitroso)": "[#6][N]=[O]",
    "N-Oxide (imine-ether)": "[#6]=[NX2][OX2][#6]",
    "N-Oxide": "[NX2]=[O]",
    "N-hydroxyl (imine)": "[#6]=[NX2][OX2H]",  
    "N-hydroxyl (generic)": "[N][OH]",         
    "N-hydroxyl (amine)": "[NX3][OX2H]",
    "Nitroso": "[N+,NX1,NX2]=O",
    "Pyridine": "n1ccccc1",
    "Pyrrole": "[nH]1cccc1",
    "Furan": "o1cccc1",
    "Thiophene": "s1cccc1",
    "Quinone": "O=C1C=CC(=O)C=C1",
    "Azo": "[#6][NX2]=[NX2][#6]",
    "Cyclopropane": "C1CC1",
    "Cyclobutane": "C1CCC1",
    "Cyclopentane": "C1CCCC1",
    "Cyclohexane": "C1CCCCC1",
    "Cyclohexene": "C1=CCCCC1",
    "Cyclohexadiene": "C1=CC=CCC1",
    "Benzene": "c1ccccc1",
    "Naphthalene": "c12ccccc1cccc2",
    "Anthracene": "c1ccc2cc3ccccc3cc2c1",
    "Phenanthrene": "c1ccc2c(c1)ccc3ccccc23",
    "Indole": "n1ccc2ccccc12",
    "Isocyanide": "[C-]#[N+]",
    "Isocyanide": "[N]#[C-]"
}

def get_functional_groups(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    mol = Chem.AddHs(mol) 
    fg_counts = defaultdict(int)

    arene_matches = set()
    for fg_name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if not pattern:
            continue
            
        matches = mol.GetSubstructMatches(pattern)
        if matches:
            if fg_name in {"Benzene", "Naphthalene", "Pyridine"}:
                for match in matches:
                    atoms = frozenset(match)
                    if atoms not in arene_matches:
                        fg_counts[fg_name] += 1
                        arene_matches.add(atoms)
            else:
                fg_counts[fg_name] += len(matches)
    
    return {k: v for k, v in fg_counts.items() if v > 0}


def detect_main_functional_groups(fg_counts: dict, smiles: str) -> dict:
    
    d = fg_counts.copy() 

    if "Naphthalene" in d:
        if "Benzene" in d:
            d["Benzene"] = max(0, d["Benzene"] - 2 * d["Naphthalene"])
    if "Anthracene" in d:
        if "Benzene" in d:
            d["Benzene"] = max(0, d["Benzene"] - 3 * d["Anthracene"])
        if "Naphthalene" in d:
            d["Naphthalene"] = max(0, d["Naphthalene"] - 2 * d["Anthracene"])
    if "Phenanthrene" in d:
        if "Benzene" in d:
            d["Benzene"] = max(0, d["Benzene"] - 3 * d["Phenanthrene"])
        if "Naphthalene" in d:
            d["Naphthalene"] = max(0, d["Naphthalene"] - 2 * d["Phenanthrene"])
    if "Indole" in d:
        if "Benzene" in d:
            d["Benzene"] = max(0, d["Benzene"] - d["Indole"])
        if "Pyrrole" in d:
            d["Pyrrole"] = max(0, d["Pyrrole"] - d["Indole"])
    if "Quinone" in d:
        if "Ketone" in d:
            d["Ketone"] = max(0, d["Ketone"] - 2 * d["Quinone"])
    if "Lactam" in d:
        for group in ["Amide", "Amine (Secondary)", "Ketone"]:
            if group in d:
                d[group] = max(0, d[group] - d["Lactam"])
    if "Peracid" in d:
        for group in ["Hydroperoxide", "Ketone"]:
            if group in d:
                d[group] = max(0, d[group] - d["Peracid"])
    if "Acyl Halide" in d:
        acyl_halide_count = d["Acyl Halide"]
        acyl_halide_substituents = {
            "Fluoroalkane": "R-CO-F",  
            "Chloroalkane": "R-CO-Cl",
            "Bromoalkane": "R-CO-Br",  
            "Iodoalkane": "R-CO-I"    
        }
        for haloalkane, acyl_type in acyl_halide_substituents.items():
            if haloalkane in d:
                d[haloalkane] = max(0, d[haloalkane] - acyl_halide_count)
        if "Ketone" in d.keys():
            d["Ketone"] = max(0, d["Ketone"] - d["Acyl Halide"])
    if "Acid Anhydride" in d:
        if "Ether" in d:
            d["Ether"] = max(0, d["Ether"] - d["Acid Anhydride"])
        if "Ester" in d:
            d["Ester"] = max(0, d["Ester"] - 2 * d["Acid Anhydride"])
        if "Ketone" in d:
            d["Ketone"] = max(0, d["Ketone"] - 2 * d["Acid Anhydride"])
    if "Lactone" in d:
        for group in ["Ester", "Ketone", "Ether"]:
            if group in d:
                d[group] = max(0, d[group] - d["Lactone"])
    if "Ester" in d:
        for group in ["Ether", "Ketone"]:
            if group in d:
                d[group] = max(0, d[group] - d["Ester"])
    if "Carboxylic Acid" in d:
        for group in ["Alcohol", "Ketone"]:
            if group in d:
                d[group] = max(0, d[group] - d["Carboxylic Acid"])
    if "Epoxide" in d and "Ether" in d:
        d["Ether"] = max(0, d["Ether"] - d["Epoxide"])
    if "Aldehyde" in d and "Ketone" in d:
        d["Ketone"] = max(0, d["Ketone"] - d["Aldehyde"])
    if "Isocyanate" in d and "Ketone" in d:
        d["Ketone"] = max(0, d["Ketone"] - d["Isocyanate"])
    
    return {k: v for k, v in d.items() if v > 0}

def count_ch_bonds(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    mol = Chem.AddHs(mol)

    
    sp3_ch = sp2_ch = sp_ch = 0
    
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'C' and not atom.IsInRing():
            h_count = sum(1 for neighbor in atom.GetNeighbors() 
                         if neighbor.GetSymbol() == 'H')
            
            has_triple = any(bond.GetBondType() == Chem.BondType.TRIPLE 
                            for bond in atom.GetBonds())
            if has_triple:
                sp_ch += h_count
                continue
                
            hybridization = atom.GetHybridization()
            if hybridization == Chem.HybridizationType.SP3:
                sp3_ch += h_count
            elif hybridization == Chem.HybridizationType.SP2:
                sp2_ch += h_count
            elif hybridization == Chem.HybridizationType.SP:
                sp_ch += h_count
    
    return {
        "sp³ C-H": sp3_ch,
        "sp² C-H": sp2_ch,
        "sp C-H": sp_ch
    }

def count_carbon_bonds(smiles):
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string.")
    
    cc_single = 0
    cc_double = 0
    cc_triple = 0
    
    for bond in mol.GetBonds():
        atom1 = bond.GetBeginAtom()
        atom2 = bond.GetEndAtom()
        
        # Only consider C-C bonds
        if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
            bond_type = bond.GetBondType()
            
            if bond_type == Chem.BondType.SINGLE:
                cc_single += 1
            elif bond_type == Chem.BondType.DOUBLE:
                cc_double += 1
            elif bond_type == Chem.BondType.TRIPLE:
                cc_triple += 1
    
    return {
        "C-C (single)": cc_single,
        "C=C (double)": cc_double,
        "C≡C (triple)": cc_triple
    }

def analyze_molecule(smiles: str) -> dict:
    fg = get_functional_groups(smiles)
    fg_main = detect_main_functional_groups(fg, smiles)
    ch_counts = count_ch_bonds(smiles)
    cc_bond_counts = count_carbon_bonds(smiles)

    combined = {}
    combined.update(fg_main)
    combined.update(ch_counts)
    combined.update(cc_bond_counts)

    return combined



""""
#Option 1
import json
with open("../data/functional_groups_ir.json") as f:
    FUNCTIONAL_GROUPS_IR = json.load(f)

#Option 2
import importlib.util
import os

relative_path = os.path.join(os.path.dirname(__file__), "../../data/dictionnary.py")
absolute_path = os.path.abspath(relative_path)

spec = importlib.util.spec_from_file_location("dictionnary", absolute_path)
dictionnary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dictionnary)

print("Available attributes in dictionnary:", dir(dictionnary)) 
FUNCTIONAL_GROUPS_IR = dictionnary.FUNCTIONAL_GROUPS_IR
components = ["Isocyanide", "Isocyanide"]
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def gaussian(x, center, intensity, width):
    """Generate a single Gaussian peak."""
    return intensity * np.exp(-((x - center) ** 2) / (2 * width ** 2))

def reconstruct_spectrum(x_axis, peaks):
    """Sum multiple Gaussian peaks."""
    y = np.zeros_like(x_axis)
    for center, intensity, width in peaks:
        y += gaussian(x_axis, center, intensity, width)
    return y

def build_and_plot_ir_spectrum(FUNCTIONAL_GROUPS_IR: dict, components: dict, common_axis=None):
    if common_axis is None:
        common_axis = np.linspace(400, 4000, 5000)

    combined_peaks = []

    for group_name, count in components.items():
        group_data = FUNCTIONAL_GROUPS_IR.get(group_name)
        if group_data:
            freqs = group_data["frequencies"]
            intensities = group_data["intensities"]
            widths = group_data["widths"]

            for f, i, w in zip(freqs, intensities, widths):
                combined_peaks.append((f, i * count, w))

    absorbance = reconstruct_spectrum(common_axis, combined_peaks)
    absorbance /= np.max(absorbance) if np.max(absorbance) > 0 else 1
    transmittance = 1 - absorbance

    plt.figure(figsize=(8, 4))
    plt.plot(common_axis, -absorbance, label="Simulated IR Spectrum")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Relative Absorbance (a.u.)")
    plt.title("Simulated IR Spectrum from Functional Groups")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return common_axis, transmittance

build_and_plot_ir_spectrum({

"C=C (double)": {
    "frequencies": [1600, 1680],
    "intensities": [35],
    "widths": [25]
},

"C≡C (triple)": {
    "frequencies": [2100, 2260],
    "intensities": [12.5],
    "widths": [25]
},  "Phenol": {
        "frequencies": [3200, 3600],
        "intensities": [75],
        "widths": [65]
    },
    "Alcohol": {
        "frequencies": [3200, 3600],
        "intensities": [75],
        "widths": [65]
    },
    "Benzene": {
        "frequencies": [3000, 3100],
        "intensities": [35],
        "widths": [10]
    },
    "sp² C-H": {
    "frequencies": [3010, 3100],  # Alkene C-H
    "intensities": [35],
    "widths": [25]
}
}
, {"C=C (double)": 3, "C≡C (triple)": 0 , "Phenol": 0,"Benzene": 0 , "Alcohol": 0, "sp² C-H": 6})