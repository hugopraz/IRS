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

[![EPFL Course](https://img.shields.io/badge/EPFL-red?style=for-the-badge)](https://edu.epfl.ch/coursebook/en/practical-programming-in-chemistry-CH-200)

## Contributors
- Ryans Chen               [![GitHub](https://img.shields.io/badge/GitHub-ryanschen0-181717.svg?style=flat&logo=github)](https://github.com/ryanschen0)
- Hugo Praz                [![GitHub](https://img.shields.io/badge/GitHub-hugopraz-181717.svg?style=flat&logo=github)](https://github.com/hugopraz)
- Anders Thomas Eggen      [![GitHub](https://img.shields.io/badge/GitHub-Anders--Eggen-181717?style=for-the-badge&logo=github)](https://github.com/Anders-Eggen) 

[![python3.10](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=orange)](https://www.python.org) [![LICENSE](https://img.shields.io/badge/License-MIT-purple.svg)](https://github.com/Flo-fllt/Projet_chem/blob/main/LICENSE.txt) 

Commit activity: [![Commit Activity](https://img.shields.io/badge/Commit%20Activity-View%20Graph-blue?style=for-the-badge&logo=github)](https://github.com/ryanschen0/IRS/graphs/commit-activity)

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ThomasCsson/MASSIVEChem)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Anaconda](https://img.shields.io/badge/Anaconda-44A833.svg?&style=for-the-badge&logo=anaconda&logoColor=white)](https://www.anaconda.com/)

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
| Data Handling | `collections`, `pandas` |
| QM Engine     | `Psi4`                  |
| Visualization | `py3Dmol`, `Matplotlib` |
| Interface     | `Streamlit`             |
| Math / Logic  | `NumPy`                 |


## 🔥 Usage


## 🛠️ Installation
Pip install
IRS can be installed using pip
```bash
pip install IRS
```

GitHub
Install via pip using the following command
```bash
pip install git+https://github.com/ryanschen0/IRS
```

Git
The package can also be installed from the GitHub repository.
Using this method, it is advised to create a CONDA environement fist:
```bash
#Open bash or terminal and type
conda create -n env.name
#Name the environment as you wish

#Activate the environment
conda activate env.name
```
Then clone the repository form github
```bash
git clone https://github.com/ryanschen0/IRS.git
cd path/to/IRS
```
Finally, install the package uisng the follwing commands
```bash
pip install -e
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
```
This method also requires the installation of ORCA (>= 5.0.2).

Sturctural Approach
```bash
rdkit (>= 2022.9.5)
matplotlib (>=3.4.0)
streamlit
pandas
```

If the installation is successful, the packages mentionned above should all be installed automatically. However, this can be verified by checking if all have been installed in the desired environnement using the following commands:

| Goal                                             | Command                      |
|-----------------------------------------------|------------------------------|
| Check if a specific package is installed      | `pip show IRS`       |
| See a list of all installed packages          | `pip list`                   |
| Search for a package in the list (Linux/macOS)| `pip list \| grep IRS`   |
| Search for a package in the list (Windows)    | `pip list \| findstr IRS`   |


## Need help?
If you encounter issues or the program doesn't work, try the following steps to troubleshoot:

1. Verify your active environment
Make sure you are working in the environment where RetroChem is installed.
```bash
# Check which Python executable is currently active
which python
```
If it's not the correct environment, acitvate it:
```bash
# Activate your conda environment
conda activate env.name
```
2. Navigate to the IRS directory
Go to the RetroChem folder to ensure you are in the right place
```bash
cd IRS
```
Confirm your current directory
```bash
pwd
```
The output should end with `/IRS`
3. Check and update IRS
```bash
pip show IRS
```
If needed, update to the latest version
```bash
pip install --upgrade IRS
```
If problems continue, try uninstalling and reinstalling IRS, specifying the desired version
```bash
pip uninstall IRS
pip install IRS==x.x.x  #replace x.x.x with the specific version desired
```
4. Update pip if necessary
Sometimes, issues may arise due to an outdated pip. Thus, to update pip:
- For virtual environments
```bash
pip install --upgrade pip
```
- For Linux or macOS systems
```bash
python3 -m pip install --upgrade pip
```