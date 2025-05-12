import os
import json
import pytest
import numpy as np
from IRS.src.irs.irs_table import (
    gaussian,
    reconstruct_spectrum,
    combine_spectra_from_peaks
)

# --- Test Data ---
SAMPLE_PEAKS = [(1000, 0.8, 50), (1500, 0.5, 30)]
SAMPLE_SPECTRA = {
    "Isocyanide": [(2100, 0.9, 40), (1200, 0.3, 20)],
    "Hydroxyl": [(3400, 0.7, 60)]
}

# --- Gaussian Function Tests ---

# Tests basic properties of the Gaussian function including symmetry and peak value
def test_gaussian():
    x = np.array([0, 1, 2])
    result = gaussian(x, center=1, intensity=1.0, sigma=0.5)
    assert result[1] == pytest.approx(1.0)
    assert result[0] == result[2]
    assert np.all(result >= 0)

# Tests handling of zero intensity input
def test_gaussian_zero_intensity():
    result = gaussian(np.linspace(0, 10, 5), 5, 0, 1)
    assert np.all(result == 0)

# Tests error handling for invalid sigma values
def test_gaussian_zero_sigma():
    with pytest.raises(ZeroDivisionError):
        gaussian(0, 0, 1, 0)

# --- Spectrum Reconstruction Tests ---

# Tests reconstruction of spectrum from peak parameters
def test_reconstruct_spectrum():
    x_axis = np.linspace(400, 4000, 5000)
    result = reconstruct_spectrum(x_axis, SAMPLE_PEAKS)
    assert result.shape == x_axis.shape
    assert np.max(result) > 0.5

# Tests spectrum reconstruction with empty peak list
def test_reconstruct_spectrum_empty_peaks():
    x_axis = np.linspace(400, 4000, 5000)
    result = reconstruct_spectrum(x_axis, [])
    assert np.all(result == 0)

# --- Spectrum Combination Tests ---

# Tests combination of spectra with duplicate components
def test_combine_spectra_from_peaks():
    components = ["Isocyanide", "Isocyanide"]
    x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, components)
    peak_idx = np.abs(x - 2100).argmin()
    assert transmittance[peak_idx] < 0.2

# Tests spectrum combination with empty component list
def test_combine_spectra_from_peaks_empty_components():
    x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, [])
    assert np.all(transmittance == 1)

# Tests handling of unknown components in spectrum combination
def test_combine_spectra_from_peaks_unknown_components():
    x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, ["Unknown"])
    assert np.all(transmittance == 1)

# Tests custom x-axis handling in spectrum combination
def test_combine_spectra_from_peaks_custom_axis():
    custom_x = np.linspace(500, 3000, 100)
    x, _ = combine_spectra_from_peaks(SAMPLE_SPECTRA, ["Isocyanide"], custom_x)
    assert np.array_equal(x, custom_x)

# --- Property-Based Tests ---

# Tests Gaussian intensity scaling with parameterized inputs
@pytest.mark.parametrize("intensity", [0.1, 0.5, 1.0])
def test_gaussian_intensity_scaling(intensity):
    x = np.linspace(0, 10, 100)
    result = gaussian(x, 5, intensity, 1)
    assert result.max() == pytest.approx(intensity)

# Tests component counting with parameterized inputs
@pytest.mark.parametrize("count", [1, 2, 5])
def test_combine_spectra_from_peaks_component_counting(count):
    components = ["Isocyanide"] * count
    x, transmittance = combine_spectra_from_peaks(SAMPLE_SPECTRA, components)
    peak_idx = np.abs(x - 2100).argmin()
    expected = 1 - min(1, count * 0.9)
    assert transmittance[peak_idx] == pytest.approx(expected, abs=0.1)

# --- Data Loading Tests ---

# Tests JSON data loading functionality
def test_functional_groups_json_loading():
    try:
        with open("../data/functional_groups_ir.json") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "Isocyanide" in data
    except FileNotFoundError:
        pytest.skip("JSON test data not available")

# Tests module-based data loading fallback
def test_dictionary_module_loading():
    try:
        import importlib.util
        module_path = os.path.join(os.path.dirname(__file__), "../../data/dictionnary.py")
        spec = importlib.util.spec_from_file_location("dictionnary", module_path)
        dictionnary = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dictionnary)
        assert hasattr(dictionnary, 'FUNCTIONAL_GROUPS_IR')
    except ImportError:
        pytest.skip("Module import test data not available")