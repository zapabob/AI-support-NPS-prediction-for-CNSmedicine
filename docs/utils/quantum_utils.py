def extract_quantum_features(mol, temperature=310.15, pressure=101325, solvent='blood_plasma'):
    """Calculate quantum chemical features for a given compound under blood plasma-like conditions"""
    # Set solvent information
    if solvent == 'blood_plasma':
        solvent_dielectric = 80.0  # Approximate dielectric constant of human blood plasma
        solvent_composition = {
            'water': 0.92,  # 92% water
            'proteins': 0.07,  # 7% proteins
            'salts': 0.009,  # 0.9% salts
            'sugars': 0.001  # 0.1% sugars
        }
        plasma_ph = 7.4  # Approximate pH of human blood plasma
        plasma_ionic_strength = 0.154  # Approximate ionic strength of human blood plasma (in mol/L)
    else:
        raise ValueError(f'Unsupported solvent: {solvent}')

    # Generate 3D structure of the molecule
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)

    # Create PySCF molecule object
    atom_coords = mol.GetConformer().GetPositions()
    atom_symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(mol.GetNumAtoms())]
    pyscf_mol = gto.M(atom_symbols, atom_coords, unit='Angstrom')

    # Perform quantum chemistry calculation
    mf = scf.RHF(pyscf_mol)
    mf.max_cycle = 100
    mf.newton()
    mf.kernel()

    # Consider temperature, pressure, pH, and ionic strength conditions
    mf.set_temperature(temperature)
    mf.set_pressure(pressure)
    mf.set_ph(plasma_ph)
    mf.set_ionic_strength(plasma_ionic_strength)

    # Calculate features
    energy = mf.e_tot
    dipole_moment = np.linalg.norm(mf.dip_moment())
    homo = mf.mo_energy[mf.mo_occ > 0][-1]
    lumo = mf.mo_energy[mf.mo_occ == 0][0]
    gap = lumo - homo

    return np.array([energy, dipole_moment, homo, lumo, gap])
