import unittest
import os
import tempfile
import numpy as np
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock

# Path Configuration
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# RDKit imports
from rdkit import Chem
from rdkit.Chem import AllChem

# Import functions to test
import src.irs.QM_combiner as qm

# Mock classes for testing
class MockMol:
    def __init__(self, charge=0, unpaired=0):
        self.charge = charge
        self.unpaired = unpaired
        self.atoms = []
        
    def GetFormalCharge(self):
        return self.charge
        
    def GetAtoms(self):
        return self.atoms
        
    def GetNumAtoms(self):
        return len(self.atoms)
        
    def GetConformer(self):
        return MockConformer()

class MockAtom:
    def __init__(self, symbol="C", idx=0, radical_electrons=0):
        self.symbol = symbol
        self.idx = idx
        self.radical_electrons = radical_electrons
        
    def GetSymbol(self):
        return self.symbol
        
    def GetIdx(self):
        return self.idx
        
    def GetNumRadicalElectrons(self):
        return self.radical_electrons

class MockConformer:
    def GetAtomPosition(self, idx):
        return MockPosition()

class MockPosition:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class TestOrcaFunctions(unittest.TestCase):


    """""
    # Tests if a valid ORCA output file is correctly parsed to extract IR frequencies and intensities
    def test_parse_orca_output_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_out = os.path.join(tmpdir, "test.out")
            with open(test_out, 'w') as f:
                f.write("""
    """""
    IR SPECTRUM
    Mode    freq    intensity
    1:      1000.0  50.0
    2:      2000.0  30.0
    """
    """"""
    """"
            freqs, inten = parse_orca_output(test_out)
            self.assertEqual(freqs, [1000.0, 2000.0])
            self.assertEqual(inten, [50.0, 30.0])
    """
    """
    # Verifies that small molecules are correctly optimized using psi4 geometry optimization
    @patch('psi4.geometry')
    def test_smiles_to_optimized_geometry_small_mol(self, mock_geo):
        mock_geo.return_value = "mock_molecule"
        mock_mol = MagicMock(spec=Chem.rdchem.Mol)
        mock_mol.GetNumAtoms.return_value = 2
        with patch('src.irs.QM_combiner.generate_3d_molecule', return_value=mock_mol):
            # Note: Fixed function call to avoid non-existent module error
            mol, rdkit_mol = smiles_to_optimized_geometry("CC", "HF/STO-3G")
            self.assertEqual(mol, "mock_molecule")
    """
    # Tests if mol_to_3dviewer correctly configures the 3D viewer with proper styling
    @patch('rdkit.Chem.MolToMolBlock')
    @patch('py3Dmol.view')
    def test_mol_to_3dviewer_configuration(self, mock_view, mock_to_molblock):
        mock_mol = MagicMock()
        mock_molblock = "MOCK MOLBLOCK"
        mock_to_molblock.return_value = mock_molblock
        mock_viewer = MagicMock()
        mock_view.return_value = mock_viewer
        
        result = qm.mol_to_3dviewer(mock_mol)
        
        mock_to_molblock.assert_called_once_with(mock_mol)
        mock_view.assert_called_once_with(width=400, height=300)
        mock_viewer.addModel.assert_called_once_with(mock_molblock, 'mol')
        mock_viewer.setStyle.assert_called_once()
        mock_viewer.setBackgroundColor.assert_called_once_with('white')
        mock_viewer.zoomTo.assert_called_once()
        self.assertEqual(result, mock_viewer)


if __name__ == '__main__':
    unittest.main()