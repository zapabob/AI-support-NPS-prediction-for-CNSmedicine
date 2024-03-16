import pandas as pd
from rdkit import Chem
from chembl_webresource_client.new_client import new_client
from sklearn.preprocessing import StandardScaler
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer

# Internal data: SMILES and IUPAC names of standard compounds
standard_compounds = {
    'D-Amphetamine': {'smiles': 'CC(N)Cc1ccccc1', 'iupac': '(R)-1-phenylpropan-2-amine'},
    'Cocaine': {'smiles': 'COC(=O)C1C(OC(=O)c2ccccc2)CC2CCC1N2C', 'iupac': 'methyl (1R,2R,3S,5S)-3-(benzoyloxy)-8-methyl-8-azabicyclo[3.2.1]octane-2-carboxylate'},
    'Methylphenidate': {'smiles': 'COC(=O)C(C1CCCCN1)c1ccccc1', 'iupac': 'methyl 2-phenyl-2-(piperidin-2-yl)acetate'},
    'LSD-25': {'smiles': 'CCN(CC)C(=O)C1CN(C)C2CC3=CNC4=CC=CC(=C34)C2=C1', 'iupac': '(6aR,9R)-N,N-diethyl-7-methyl-4,6,6a,7,8,9-hexahydroindolo[4,3-fg]quinoline-9-carboxamide'},
    'MDMA': {'smiles': 'CC(NC)Cc1ccc2c(c1)OCO2', 'iupac': '1-(benzo[d][1,3]dioxol-5-yl)-N-methylpropan-2-amine'},
    '2C-B': {'smiles': 'CC(Cc1ccc2c(c1)OCO2)NC', 'iupac': '2-(4-bromo-2,5-dimethoxyphenyl)ethanamine'},
    'Psilocybin': {'smiles': 'CN(C)CCc1c[nH]c2cccc(OP(=O)(O)O)c12', 'iupac': '3-[2-(dimethylamino)ethyl]-1H-indol-4-yl dihydrogen phosphate'},
    'Mescaline': {'smiles': 'COc1cc(CC(N)=O)cc2c1occc2', 'iupac': '3,4,5-trimethoxyphenethylamine'}
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
