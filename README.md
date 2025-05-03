![Project Logo](assets/banner.png)

![Coverage Status](assets/coverage-badge.svg)

<h1 align="center">
IRS
</h1>

<br>


## Infra-Red Simulator (IRS)
This project focuses on the simulation of IR spectra.
The project has two functionalities, giving two different approaches.

The first one is the simualtion of IR spectra using Psi4 and ORCA, two different quantum mechanical calculation packages.

## Theoretical Background of Infra-Red Spectroscopy
QM Calculations:
This approach uses first principle quantum mechanics to simulate an IR spectrum, using the following approximations taken by the Psi4 package:
- Molecule is in Gas Phase at T=0K 
- Harmonic Approximation for Frequency Calculations

The vibrational frequencies are calculated by assuming the lowest harmonic energy potential. The Psi4 package then computes the Hessian matrix, which is diagonalized to obtain normal mode frquencies. The IR intensities are then computed by analytically calculating the change of the dipole moment in respect of the vibrational motion.








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

### Run tests and coverage

```
(conda_env) $ pip install tox
(conda_env) $ tox
```



