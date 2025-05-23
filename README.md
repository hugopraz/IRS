![Project Logo](assets/irs_logo.png)

<h1 align="center">
IRS
</h1>

<br>

## Infra-Red Simulator (IRS)
IRS – Infra-Red Simulator – is a Python-based application developed for the simulation and visualization of Infra-Red (IR) spectra of molecules. It provides a web-based interface for converting molecular names or SMILES strings into fully optimized 3D structures, performing vibrational analysis via quantum chemistry packages, and plotting the corresponding IR spectrum.

The project provides two main functionalities, offering two different approaches.
The first simulates IR spectra using Psi4 and ORCA, two quantum mechanical calculation packages. The second, a structural approach, generates an approximate IR spectrum by analyzing the molecular structure to identify key functional groups, C–H bonds (classified by hybridization, e.g., sp³ C–H), and C–C bonds (e.g., C=C). The characteristic absorption peaks for each feature are then combined to construct the overall spectrum.
<div style="text-align: center;">
  <a href="https://edu.epfl.ch/coursebook/en/practical-programming-in-chemistry-CH-200">
    <img src="https://img.shields.io/badge/EPFL-red?style=for-the-badge" alt="EPFL Course">
  </a>
</div>
## Contributors
<div align="center">

| Name                  | GitHub                                  |
|-----------------------|------------------------------------------|
| Ryans Chen            | ![GitHub](https://img.shields.io/badge/GitHub-ryanschen0-181717.svg?style=flat&logo=github) |
| Hugo Praz             | ![GitHub](https://img.shields.io/badge/GitHub-hugopraz-181717.svg?style=flat&logo=github) |
| Anders Thomas Eggen   | ![GitHub](https://img.shields.io/badge/GitHub-Anders--Eggen-181717?style=flat&logo=github) |

</div>

<p align="center">
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=orange" alt="Python 3.10">
  </a>
  <a href="https://github.com/Flo-fllt/Projet_chem/blob/main/LICENSE.txt">
    <img src="https://img.shields.io/badge/License-MIT-purple.svg" alt="MIT License">
  </a>
</p>

<p align="center">
  <strong>Commit activity:</strong> <a href="https://github.com/ryanschen0/IRS/graphs/commit-activity">link</a>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" alt="Python">
  </a>
  <a href="https://www.anaconda.com/">
    <img src="https://img.shields.io/badge/Anaconda-44A833.svg?&style=for-the-badge&logo=anaconda&logoColor=white" alt="Anaconda">
  </a>
</p>


## Theoretical Background of Infra-Red Spectroscopy
QM calculations using Psi4:
This approach uses first principle quantum mechanics to simulate an IR spectrum, using the following approximations taken by the Psi4 package:
- Molecule is in Gas Phase at T=0K 
- Harmonic Approximation for Frequency Calculations
- Born–Oppenheimer approximation (separating electronic and nuclear motion)

The vibrational frequencies are calculated by assuming the lowest harmonic energy potential. The Psi4 package then computes the Hessian matrix, which is diagonalized to obtain normal mode frequencies. The IR intensities are then computed by analytically calculating the change of the dipole moment in respect of the vibrational motion.

QM Calculations using ORCA: 
This approach simulates an IR spectrum similarly to the Psi4 method, relying on Density Functional Theory (DFT) as implemented in the ORCA package. The vibrational frequencies are computed under the same approximations as in the Psi4 package. As ORCA uses different integral libraries and optimization schemes than Psi4, slight variations in intensities or frequencies are expected, especially for large molecules.

Structural approach:
This method relies on an empirical, rule-based approach to approximate IR spectra by identifying key molecular features through three distinct strategies. First, functional groups are detected using SMARTS-based substructure matching, enabling the recognition of characteristic moieties such as alcohols, ketones, and esters, each associated with specific IR absorption bands. Second, the classification of acyclic C–H bonds is performed by analyzing the hybridization state (sp³, sp², sp) of the carbon atom to which the hydrogen is attached, as these differences influence vibrational stretching frequencies. Finally, carbon–carbon bonding patterns, including single, double, and triple bonds, are counted to account for their respective spectral contributions. By combining these structural insights, the method constructs a composite IR spectrum that reflects the vibrational fingerprint of the molecule.


## Stack 
<div align="center">

| Component     | Library                 |
| ------------- | ----------------------- |
| Molecular Input/Output, Substructure Matching, and Molecular Parsing | `PubChemPy`, `RDKit`    |
| Data Handling | `collections`, `pandas` |
| QM Engine     | `Psi4`                  |
| Visualization | `py3Dmol`, `Matplotlib` |
| Interface     | `Streamlit`             |
| Math / Logic  | `NumPy`                 |

</div>



## 🔥 Usage
Once the IR Spectrum Simulator page is open, the engine is ready for use. Below is a step-by-step example using ethanol to demonstrate how to use the web interface.

<p align="center">
  <img src="assets/README-image-1.png" alt="Web page picture 1"/>
</p>

By clicking the arrow, the following tab will open.

<p align="center">
  <img src="assets/README-image-2.png" alt="Tab explanation"/>
</p>

In the case of ORCA, additional inputs for the ORCA Executable Path and Output Directory need to be filled out, as shown below.

<p align="center">
  <img src="assets/README-image-3.png" alt="Tab of ORCA explanation"/>
</p>

In addition, a 3D molecular structure generator is included, which displays the total number of atoms, the molecular charge, and the spin multiplicity of the compound, as shown below.

<p align="center">
  <img src="assets/README-image-4.png" alt="3D molecule visualizer"/>
</p>

Finally, to generate the IR spectrum, the following button must be clicked.

<p align="center">
  <img src="assets/README-image-5.png" alt="IR spectrum Example"/>
</p>

## 🛠️ Installation
Installation via pip
irspectrum-sim can be installed using pip
```bash
pip install irspectrum-sim
```

Installation from GitHub
Install via pip using the following command:
```bash
pip install git+https://github.com/ryanschen0/IRS
```

The package can also be installed from the GitHub repository.
Using this method, it is advised to create a CONDA environement fist:
```bash
#Open bash or terminal and type
conda create -n env.name
#Name the environment as you wish

#Activate the environment
conda activate env.name
```
Then clone the repository from github
```bash
git clone https://github.com/ryanschen0/IRS.git
cd path/to/IRS
```
Finally, install the package uisng the following commands
```bash
pip install -e .
```


## 📚 Requirements
The package runs on Python 3.10 but supports python 3.9. However, it requires several other packages as well.

QM Approach: Psi4
```bash
rdkit (>= 2022.9.5)
Psi4
Matplotlib
numpy (>=1.21.0, <2.0.0)
streamlit
```

QM Approach: ORCA
```bash
rdkit (>= 2022.09.1)
numpy (>=1.21.0, <2.0.0)
matplotlib (>=3.4.0)
subprocess
streamlit
```
This method also requires the installation of ORCA (>= 5.0.2).

Structural Approach
```bash
rdkit (>= 2022.9.5)
matplotlib (>=3.4.0)
streamlit
pandas
```

## Need help?
If you encounter issues or the program doesn't work, try the following steps to troubleshoot:

1. Verify your active environment
Make sure you are working in the environment where irspectrum-sim is installed.
```bash
# Check which Python executable is currently active
which python
```
If it's not the correct environment, activate it:
```bash
# Activate your conda environment
conda activate env.name
```
2. Navigate to the IRS directory
Go to the IRS folder to ensure you are in the right place
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
pip show irspectrum-sim
```
If needed, update to the latest version
```bash
pip install --upgrade irspectrum-sim
```
If problems continue, try uninstalling and reinstalling irspectrum-sim, specifying the desired version
```bash
pip uninstall irspectrum-sim
pip install irspectrum-sim==x.x.x  #replace x.x.x with the specific version desired
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

## Data Sources

The following sources were used to generate the `dict_fg_IR_data.json` dictionary. 

- [NC State Organic Chemistry Textbook](https://ncstate.pressbooks.pub/organicchem/chapter/infrared-spectra-of-some-common-functional-groups/). Accessed May 23, 2025.
- [this IR spectrum table](https://studylib.net/doc/25856154/ir-spectrum-table) on StudyLib. Accessed May 23, 2025.
- [LibreTexts Chemistry](https://chem.libretexts.org/Courses/SUNY_Oneonta/Chem_322_Lecture_Content/05%3A_Organohalogen_and_Organometallic_Compounds/5.03%3A_Spectroscopic_Properties). Accessed May 23, 2025.
- [SDBS (Spectral Database for Organic Compounds)](https://sdbs.db.aist.go.jp/SearchInformation.aspx), maintained by AIST Japan. Accessed May 23, 2025.