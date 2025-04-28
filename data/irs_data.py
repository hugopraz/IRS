import os
from jcamp import jcamp_readfile
import numpy as np
def load_ir_spectra_from_folder(folder_path):
    spectra_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".jdx"):
            file_path = os.path.join(folder_path, filename)
            data = jcamp_readfile(file_path)

            name = data.get("title", os.path.splitext(filename)[0]).strip()
            x = np.array(data["x"])
            y = np.array(data["y"])
            
            # Scale y-values if YFACTOR exists
            yfactor = float(data.get("yfactor", 1))
            y = y * yfactor

            # Check unit
            y_units = data.get("yunits", "").lower()
            if "absorbance" in y_units:
                absorbance = y
                transmittance = 10 ** (-absorbance) 
            elif "transmittance" in y_units:
                transmittance = y
            else:
                print(f"⚠️ Unknown Y-units in {filename}. Defaulting to raw y-values.")
                transmittance = y

            spectra_dict[name] = {
                "wavenumbers": x,
                "transmittance": transmittance,
            }
    return spectra_dict
spectra_dict = load_ir_spectra_from_folder("IRS/data/irs_jpr")
print(spectra_dict)