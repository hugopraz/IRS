import numpy as np
def combine_spectra(spectra_dict, component_names, common_axis=None):
    if common_axis is None:
        first = spectra_dict[component_names[0]]
        common_axis = first["wavenumbers"]

    total_absorbance = np.zeros_like(common_axis, dtype=float)

    for name in component_names:
        spec = spectra_dict[name]
        interp_trans = np.interp(common_axis, spec["wavenumbers"][::-1], spec["transmittance"][::-1])

        absorbance = -np.log10(interp_trans / 100)
        total_absorbance += absorbance

    combined_transmittance = 10 ** (-total_absorbance) * 100
    return common_axis, combined_transmittance
