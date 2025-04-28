import numpy as np
import matplotlib.pyplot as plt

class IRSpectrum:
    def __init__(self, wavenumbers: np.ndarray, transmittance: np.ndarray, title: str = "IR Spectrum"):
        if wavenumbers.shape != transmittance.shape:
            raise ValueError("Wavenumbers and transmittance must have the same shape.")
        
        self.wavenumbers = wavenumbers
        self.transmittance = transmittance
        self.title = title

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.wavenumbers, self.transmittance, color='black')
        plt.gca().invert_xaxis() 
        plt.title(self.title)
        plt.xlabel("Wavenumber (cm⁻¹)")
        plt.ylabel("Transmittance (%)")
        plt.tight_layout()
        plt.grid(True)
        plt.show()


from data.irs_data import *
#Exemple:
wavenumbers = np.linspace(4000, 400, 1600)
transmittance = 100 - 20*np.exp(-((wavenumbers - 1700)**2)/(2*40**2)) 

spectrum = IRSpectrum(spectra_dict["wavenumber"][0], spectra_dict["transmittance"][0], title="Simulated IR Spectrum")
spectrum.plot()