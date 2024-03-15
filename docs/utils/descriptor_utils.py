from rdkit.Chem import Descriptors

def calculate_descriptors(mol):
    """Calculate all available descriptors for a given compound"""
    descriptors = []

    # Add all available descriptors from RDKit
    descriptor_funcs = [x for x in dir(Descriptors) if x.startswith('_')]
    for desc_func in descriptor_funcs:
        func = getattr(Descriptors, desc_func)
        try:
            descriptor_value = func(mol)
            descriptors.append(descriptor_value)
        except Exception as e:
            print(f"Error calculating descriptor {desc_func}: {e}")

    return descriptors

def smiles_to_graph(smiles):
    graph = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer())
    return graph

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train, validation, and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=random_state)

    train_data = list(zip(X_train, y_train))
    val_data = list(zip(X_val, y_val))
    test_data = list(zip(X_test, y_test))

    return train_data, val_data, test_data
