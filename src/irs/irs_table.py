import numpy as np
import matplotlib.pyplot as plt
import os
from collections import Counter

#Gaussian and Spectrum Functions similar to ones in ir_ORCA

def gaussian(x, center, intensity, sigma):
    return intensity * np.exp(-((x - center) ** 2) / (2 * sigma ** 2))

def reconstruct_spectrum(x, peaks):
    y = np.zeros_like(x)
    for center, intensity, sigma in peaks:
        y += gaussian(x, center, intensity, sigma)
    return y

#Combine values in an IR spectrum with transimttance and multiply multiple functional groups
def combine_spectra_from_peaks(spectra_dict, component_list, common_axis=None):
    if common_axis is None:
        common_axis = np.linspace(400, 4000, 5000)

    combined_absorbance = np.zeros_like(common_axis)

    counts = Counter(component_list)

    for name, count in counts.items():
        peaks = spectra_dict.get(name)
        if peaks:
            scaled_peaks = [(center, intensity * count, sigma) for center, intensity, sigma in peaks]
            combined_absorbance += reconstruct_spectrum(common_axis, scaled_peaks)

    max_absorbance = np.max(combined_absorbance)
    if max_absorbance > 0:
        combined_absorbance /= max_absorbance
    transmittance = 1 - combined_absorbance

    return common_axis, transmittance


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





#Generate and Zoom 
x, transmittance = combine_spectra_from_peaks(FUNCTIONAL_GROUPS_IR, components)

# Determine zoom range from actual centers
all_centers = [center for comp in components for center, *_ in FUNCTIONAL_GROUPS_IR.get(comp, [])]
if all_centers:
    xmin = max(400, min(all_centers) - 150)
    xmax = min(4000, max(all_centers) + 150)
else:
    xmin, xmax = 400, 4000  # fallback range

# Plots
plt.figure(figsize=(10, 6))
plt.plot(x, transmittance, color='black', label="Combined Spectrum")
plt.xlim(xmax, xmin) 
plt.ylim(1.05, 0)    
plt.title("Combined IR Spectrum")
plt.xlabel("Wavenumber (cm⁻¹)")
plt.ylabel("Transmittance (normalized)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()