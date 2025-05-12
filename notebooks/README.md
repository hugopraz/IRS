![Project Logo](assets/irs_logo.png)

![Coverage Status](assets/coverage-badge.svg)

<h1 align="center">
IRS
</h1>

<br>

## Infra-Red Simulator (IRS)
IRS – Infra-Red Simulator – is a Python-based application developed for the simulation and visualization of Infra-Red (IR) spectra of molecules. It provides a web-based interface for converting molecular names or SMILES strings into fully optimized 3D structures, performing vibrational analysis via quantum chemistry packages, and plotting the corresponding IR spectrum.

The project has two functionalities, giving two different approaches.
The first one is the simulation of IR spectra using Psi4 and ORCA, two different quantum mechanical calculation packages. The second, a structural approach, takes a molecular structure and generates an approximate IR spectrum by identifying key functional groups, C–H bonds (classified by hybridization, e.g., sp³ C–H), and C–C bonds (e.g., C=C). Characteristic absorption peaks for each are combined to construct the overall spectrum. 

## Theoretical Background of Infra-Red Spectroscopy
QM Calculations using Psi4:
This approach uses first principle quantum mechanics to simulate an IR spectrum, using the following approximations taken by the Psi4 package:
- Molecule is in Gas Phase at T=0K 
- Harmonic Approximation for Frequency Calculations
- Born–Oppenheimer approximation (separating electronic and nuclear motion)

The vibrational frequencies are calculated by assuming the lowest harmonic energy potential. The Psi4 package then computes the Hessian matrix, which is diagonalized to obtain normal mode frquencies. The IR intensities are then computed by analytically calculating the change of the dipole moment in respect of the vibrational motion.

QM Calculations using ORCA: 
This approach simulates an IR spectra similarly to the Psi4 method, relying on Density Functional Theory (DFT) as implemented in the ORCA package. The vibrational frequencies are computed under the same approximations as in the Psi4 package. As ORCA uses different integral libraries and optimization schemes than Psi4, slight variations in intensities or frequencies are expected, especially in the case of a large molecule.

Structural approach:
This method relies on an empirical, rule-based approach to approximate IR spectra by identifying key molecular features through three distinct strategies. First, functional groups are detected using SMARTS-based substructure matching, enabling the recognition of characteristic moieties such as alcohols, ketones, and esters, each associated with specific IR absorption bands. Second, the classification of acyclic C–H bonds is performed by analyzing the hybridization state (sp³, sp², sp) of the carbon atom to which the hydrogen is attached, as these differences influence vibrational stretching frequencies. Finally, carbon–carbon bonding patterns, including single, double, and triple bonds, are counted to account for their respective spectral contributions. By combining these structural insights, the method constructs a composite IR spectrum that reflects the vibrational fingerprint of the molecule.
## Stack 

| Component     | Library                 |
| ------------- | ----------------------- |
| Molecular Input/Output, Substructure Matching & Molecular Parsing | `PubChemPy`, `RDKit`    |
| Data Handling | `collections` |
| QM Engine     | `Psi4`                  |
| Visualization | `py3Dmol`, `Matplotlib` |
| Interface     | `Streamlit`             |
| Math / Logic  | `NumPy`                 |

## 🔥 Usage
The core function of this package takes a molecule as input, provided as a SMILES string, IUPAC name, or structural drawing, and displays its corresponding infrared (IR) spectrum. This IR spectrum can be generated using  
```python
from mypackage import main_func

# One line to rule them all
result = main_func(data)
```

This usage example shows how to quickly leverage the package's main functionality with just one line of code (or a few lines of code). 
After importing the `main_func` (to be renamed by you), you simply pass in your `data` and get the `result` (this is just an example, your package might have other inputs and outputs). 
Short and sweet, but the real power lies in the detailed documentation.


## 🛠️ Installation
Pip install
IRS can be installed using pip
```bash
pip install IRS
```

GitHub
The package can also be installed from the GitHub repository via pip using the following command
```bash
pip install git+https://github.com/ryanschen0/IRS
```

Git
Clone the repository form github
```bash
git clone https://github.com/ryanschen0/IRS.git
cd path/to/IRS
```

Install the package
```bash
pip install -e
```

Initialize Git (only for the first time). 

Note: You should have create an empty repository on `https://github.com:hugopraz/IRS`.

```
git init
git add * 
git add .*
git commit -m "Initial commit" 
git branch -M main
git remote add origin git@github.com:hugopraz/IRS.git 
git push -u origin main
```

Then add and commit changes as usual. 

To install the package, run

```
(irs) $ pip install -e ".[test,doc]"
```

## 📚 Requirements
The package runs on python 3.10 but supports python 3.9. However, it requires several other packages as well.

QM Approach: Psi4
```bash
rdkit (>= 2022.9.5)
Psi4
Matplotlib
NumPy
```

QM Approach: ORCA
```bash
rdkit (>= 2022.09.1)
numpy (>=1.21.0, <2.0.0)
matplotlib (>=3.4.0)
subprocess
os
sys
```
This method also requires the installation of ORCA (>= 5.0.2).

Sturctural Approach
```bash
rdkit (>= 2022.9.5)
collections
```

If the installation is successful, the packages mentionned above should all be installed automatically. However, this can be verified by checking if all have been installed in the desired environnement using the following commands:

| Goal                                             | Command                      |
|-----------------------------------------------|------------------------------|
| Check if a specific package is installed      | `pip show IRS`       |
| See a list of all installed packages          | `pip list`                   |
| Search for a package in the list (Linux/macOS)| `pip list \| grep IRS`   |
| Search for a package in the list (Windows)    | `pip list \| findstr IRS`   |





### Run tests and coverage

```
(conda_env) $ pip install tox
(conda_env) $ tox
```