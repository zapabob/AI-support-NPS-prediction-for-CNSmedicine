from rdkit.Chem import Descriptors

def calculate_descriptors(mol):
    """Calculate descriptors for a given compound"""
    descriptors = []
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.HeavyAtomCount(mol))
    # Add other descriptors...
    return descriptors
