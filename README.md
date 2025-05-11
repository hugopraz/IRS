![Project Logo](assets/banner.png)

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
QM Calculations:
This approach uses first principle quantum mechanics to simulate an IR spectrum, using the following approximations taken by the Psi4 package:
- Molecule is in Gas Phase at T=0K 
- Harmonic Approximation for Frequency Calculations

The vibrational frequencies are calculated by assuming the lowest harmonic energy potential. The Psi4 package then computes the Hessian matrix, which is diagonalized to obtain normal mode frquencies. The IR intensities are then computed by analytically calculating the change of the dipole moment in respect of the vibrational motion.

Strucural approach:
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

```python
from mypackage import main_func

# One line to rule them all
result = main_func(data)
```

This usage example shows how to quickly leverage the package's main functionality with just one line of code (or a few lines of code). 
After importing the `main_func` (to be renamed by you), you simply pass in your `data` and get the `result` (this is just an example, your package might have other inputs and outputs). 
Short and sweet, but the real power lies in the detailed documentation.

## 👩‍💻 Installation

Create a new environment, you may also give the environment a different name. 

```
conda create -n irs python=3.10 
```

```
conda activate irs
(conda_env) $ pip install .
```

If you need jupyter lab, install it 

```
(irs) $ pip install jupyterlab
```

## 🛠️ Development installation

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
The package runs on python 3.10 but supports python 3.9. However, it requires several other packages aswell.

QM Approach: Psi4
```bash
rdkit (version 2022.9.5)
Psi4
Matplotlib
NumPy
```

QM Approach: ORCA
```bash

```

Sturctural Approach
```bash
rdkit (version 2022.9.5)
collections
```

If the installation is succesfull, the packages mentionned above should all be installed automatically. However, this can be verified






### Run tests and coverage

```
(conda_env) $ pip install tox
(conda_env) $ tox
```



