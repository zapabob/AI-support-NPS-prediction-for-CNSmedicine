import pandas as pd
from rdkit import Chem
from chembl_webresource_client.new_client import new_client
from sklearn.preprocessing import StandardScaler
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

# Internal data: SMILES of standard compounds
standard_compounds = {
    'D-Amphetamine': 'CC(N)Cc1ccccc1',
    'Cocaine': 'COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C',
    'Methylphenidate': 'COC(=O)C(C1CCCCN1)c1ccccc1',
    'LSD-25': 'CCN(CC)C(=O)C1CN(C)C2CC3=CNC4=CC=CC(=C34)C2=C1',
    'MDMA': 'CC(NC)Cc1ccc2c(c1)OCO2',
    '2C-B': 'CC(Cc1ccc2c(c1)OCO2)NC',
    'Psilocybin': 'CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12',
    'Mescaline': 'COc1cc(CC(=O)O)cc2c1occc2'
}

def load_chembl_data(target_ids):
    """Load data from ChEMBL for specified target IDs"""
    activities = new_client.activity
    res = activities.filter(target_chembl_id__in=target_ids).filter(standard_type="IC50")
    df = pd.DataFrame.from_records(res)
    df['pIC50'] = -np.log10(df['standard_value'])
    return df

def normalize_data(data):
    """Normalize the data"""
    scaler = StandardScaler()
    data['normalized_pIC50'] = scaler.fit_transform(data[['pIC50']])
    return data, scaler

def prepare_data(normalized_data):
    """Prepare train, validation, and test data"""
    X = normalized_data['canonical_smiles'].apply(lambda smiles: smiles_to_graph(smiles))
    y = normalized_data['normalized_pIC50']
    return X, y

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
