from rdkit.Chem import Descriptors

def calculate_descriptors(mol):
    """Calculate descriptors for a given compound"""
    descriptors = []

    # Basic descriptors
    descriptors.append(Descriptors.MolWt(mol))  # Molecular weight
    descriptors.append(Descriptors.HeavyAtomCount(mol))  # Number of heavy atoms
    descriptors.append(Descriptors.NHOHCount(mol))  # Number of hydroxyl groups
    descriptors.append(Descriptors.NOCount(mol))  # Number of nitrogen-oxygen groups
    descriptors.append(Descriptors.NumHAcceptors(mol))  # Number of hydrogen bond acceptors
    descriptors.append(Descriptors.NumHDonors(mol))  # Number of hydrogen bond donors
    descriptors.append(Descriptors.NumHeteroatoms(mol))  # Number of heteroatoms
    descriptors.append(Descriptors.NumRotatableBonds(mol))  # Number of rotatable bonds
    descriptors.append(Descriptors.NumValenceElectrons(mol))  # Number of valence electrons
    descriptors.append(Descriptors.TPSA(mol))  # Topological polar surface area

    # Lipinski descriptors
    descriptors.append(Descriptors.MolLogP(mol))  # LogP value
    descriptors.append(Descriptors.MolMR(mol))  # Molar refractivity

    # Charge descriptors
    descriptors.append(Descriptors.MaxPartialCharge(mol))  # Maximum partial charge
    descriptors.append(Descriptors.MinPartialCharge(mol))  # Minimum partial charge

    # Geometric descriptors
    descriptors.append(Descriptors.PMI1(mol))  # First principal moment of inertia
    descriptors.append(Descriptors.PMI2(mol))  # Second principal moment of inertia
    descriptors.append(Descriptors.PMI3(mol))  # Third principal moment of inertia
    descriptors.append(Descriptors.RadiusOfGyration(mol))  # Radius of gyration

    # Topological descriptors
    descriptors.append(Descriptors.BalabanJ(mol))  # Balaban J value
    descriptors.append(Descriptors.BertzCT(mol))  # Bertz CT value
    descriptors.append(Descriptors.Chi0(mol))  # Connectivity index chi-0
    descriptors.append(Descriptors.Chi1(mol))  # Connectivity index chi-1
    descriptors.append(Descriptors.HallKierAlpha(mol))  # Hall-Kier alpha value
    descriptors.append(Descriptors.Kappa1(mol))  # Kappa shape index 1
    descriptors.append(Descriptors.Kappa2(mol))  # Kappa shape index 2
    descriptors.append(Descriptors.Kappa3(mol))  # Kappa shape index 3

    return descriptors
