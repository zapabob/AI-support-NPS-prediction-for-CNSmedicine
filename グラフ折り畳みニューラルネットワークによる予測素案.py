import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors, AllChem
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import optuna
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats

# Check if ROCm is available, otherwise use CUDA if available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using CUDA')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using MPS')
else:
    device = torch.device('cpu')
    print('Using CPU')

# Load data from ChEMBL
target_ids = ['CHEMBL238', 'CHEMBL228', 'CHEMBL224', 'CHEMBL240']
activities = new_client.activity
res = activities.filter(target_chembl_id__in=target_ids).filter(standard_type="IC50")
df = pd.DataFrame.from_records(res)

# Calculate pIC50 values
df['pIC50'] = -np.log10(df['standard_value'])

# Normalize IC50 values across different experimental conditions
scaler = StandardScaler()
df['normalized_pIC50'] = scaler.fit_transform(df[['pIC50']])

# Extract molecular descriptors, fingerprints, and 3D descriptors using RDKit
def extract_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    descriptors = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        GraphDescriptors.BertzCT(mol),
        GraphDescriptors.Chi0(mol),
        GraphDescriptors.Chi1(mol),
        GraphDescriptors.Chi2n(mol),
        GraphDescriptors.Chi3n(mol),
        GraphDescriptors.Chi4n(mol),
        GraphDescriptors.HallKierAlpha(mol),
        GraphDescriptors.Kappa1(mol),
        GraphDescriptors.Kappa2(mol),
        GraphDescriptors.Kappa3(mol),
    ]
    
    fingerprints = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024))
    
    conformer = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    mol_3d = Chem.MolToMolBlock(mol)
    
    return descriptors + fingerprints + [mol_3d]

df['features'] = df['canonical_smiles'].apply(extract_features)
df = df[df['features'].notnull()]

# Split data into features (X) and targets (y)
X = pd.DataFrame(df['features'].tolist())
y_dat = df[df['target_chembl_id'] == 'CHEMBL238']['normalized_pIC50']
y_net = df[df['target_chembl_id'] == 'CHEMBL228']['normalized_pIC50']
y_5ht2a = df[df['target_chembl_id'] == 'CHEMBL1921']['normalized_pIC50']
y_herg = df[df['target_chembl_id'] == 'CHEMBL240']['normalized_pIC50']

# Split data into train, validation, and test sets
X_train, X_test, y_dat_train, y_dat_test, y_net_train, y_net_test, y_5ht2a_train, y_5ht2a_test, y_herg_train, y_herg_test = train_test_split(
    X, y_dat, y_net, y_5ht2a, y_herg, test_size=0.2, random_state=42
)
X_train, X_val, y_dat_train, y_dat_val, y_net_train, y_net_val, y_5ht2a_train, y_5ht2a_val, y_herg_train, y_herg_val = train_test_split(
    X_train, y_dat_train, y_net_train, y_5ht2a_train, y_herg_train, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_dat_train_tensor = torch.tensor(y_dat_train.values, dtype=torch.float32).to(device)
y_dat_val_tensor = torch.tensor(y_dat_val.values, dtype=torch.float32).to(device)
y_dat_test_tensor = torch.tensor(y_dat_test.values, dtype=torch.float32).to(device)

# Graph Convolutional Neural Network (GNN) model
class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = nn.ReLU()(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

# Training function for GNN
def train_gnn(model, loader, optimizer, criterion, device):
    model.train()
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

# Evaluation function for GNN
def evaluate_gnn(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            loss = criterion(output, data.y)
            total_loss += loss.item()
    return total_loss / len(loader)

# Optuna objective function for GNN hyperparameter optimization
def objective_gnn(trial):
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    
    model = GNN(X_train.shape[1], hidden_dim, 1, dropout).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_data_list = [Data(x=torch.tensor(X_train.iloc[i].values, dtype=torch.float32), 
                            y=torch.tensor(y_dat_train.iloc[i], dtype=torch.float32)) 
                       for i in range(len(X_train))]
    val_data_list = [Data(x=torch.tensor(X_val.iloc[i].values, dtype=torch.float32), 
                          y=torch.tensor(y_dat_val.iloc[i], dtype=torch.float32)) 
                     for i in range(len(X_val))]
    train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data_list, batch_size=32)
    
    for epoch in range(100):
        train_gnn(model, train_loader, optimizer, criterion, device)
    
    val_loss = evaluate_gnn(model, val_loader, criterion, device)
    return val_loss

# Optuna objective function for Random Forest hyperparameter optimization
def objective_rf(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 500)
    max_depth = trial.suggest_int('max_depth', 2, 32)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
    
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                  min_samples_split=min_samples_split,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=42)
    model.fit(X_train, y_dat_train)
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_dat_val, y_pred)
    return mse

# Optimize GNN hyperparameters using Optuna
study_gnn = optuna.create_study(direction='minimize')
study_gnn.optimize(objective_gnn, n_trials=100)
best_params_gnn = study_gnn.best_params

# Optimize Random Forest hyperparameters using Optuna
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=100)
best_params_rf = study_rf.best_params

# Train GNN model with best hyperparameters and evaluate with cross-validation
def train_and_evaluate_gnn_cv(X, y, n_splits=5, hidden_dim=None, dropout=None, learning_rate=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        train_data_list = [Data(x=torch.tensor(X_train.iloc[i].values, dtype=torch.float32), 
                                y=torch.tensor(y_train.iloc[i], dtype=torch.float32)) 
                           for i in range(len(X_train))]
        test_data_list = [Data(x=torch.tensor(X_test.iloc[i].values, dtype=torch.float32), 
                               y=torch.tensor(y_test.iloc[i], dtype=torch.float32)) 
                          for i in range(len(X_test))]
        train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_data_list, batch_size=32)
        
        model = GNN(X_train.shape[1], hidden_dim, 1, dropout).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        for epoch in range(100):
            train_gnn(model, train_loader, optimizer, criterion, device)
        
        test_loss = evaluate_gnn(model, test_loader, criterion, device)
        cv_scores.append(test_loss)
    
    return np.mean(cv_scores)

gnn_cv_score = train_and_evaluate_gnn_cv(X, y_dat, hidden_dim=best_params_gnn['hidden_dim'], 
                                         dropout=best_params_gnn['dropout'], 
                                         learning_rate=best_params_gnn['learning_rate'])

# Train Random Forest model with best hyperparameters and evaluate with cross-validation
def train_and_evaluate_rf_cv(X, y, n_splits=5, n_estimators=None, max_depth=None, 
                             min_samples_split=None, min_samples_leaf=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth,
                                      min_samples_split=min_samples_split,
                                      min_samples_leaf=min_samples_leaf,
                                      random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        cv_scores.append(mse)
    
    return np.mean(cv_scores)

rf_cv_score = train_and_evaluate_rf_cv(X, y_dat, n_estimators=best_params_rf['n_estimators'], 
                                       max_depth=best_params_rf['max_depth'],
                                       min_samples_split=best_params_rf['min_samples_split'], 
                                       min_samples_leaf=best_params_rf['min_samples_leaf'])

# Train final GNN model with best hyperparameters
best_gnn_model = GNN(X_train.shape[1], best_params_gnn['hidden_dim'], 1, best_params_gnn['dropout']).to(device)
optimizer = optim.Adam(best_gnn_model.parameters(), lr=best_params_gnn['learning_rate'])
criterion = nn.MSELoss()

train_data_list = [Data(x=torch.tensor(X_train.iloc[i].values, dtype=torch.float32), 
                        y=torch.tensor(y_dat_train.iloc[i], dtype=torch.float32)) 
                   for i in range(len(X_train))]
train_loader = DataLoader(train_data_list, batch_size=32, shuffle=True)

for epoch in range(100):
    train_gnn(best_gnn_model, train_loader, optimizer, criterion, device)

# Train final Random Forest model with best hyperparameters
best_rf_model = RandomForestRegressor(n_estimators=best_params_rf['n_estimators'], 
                                      max_depth=best_params_rf['max_depth'],
                                      min_samples_split=best_params_    self.create_widgets()

def create_widgets(self):
    self.label = ttk.Label(self, text="Enter IUPAC name or SMILES:")
    self.label.pack(pady=10)
    
    self.entry = ttk.Entry(self, width=50)
    self.entry.pack()
    
    self.predict_button = ttk.Button(self, text="Predict", command=self.predict)
    self.predict_button.pack(pady=10)
    
    self.result_frame = ttk.LabelFrame(self, text="Prediction Results")
    self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    
    self.result_text = tk.Text(self.result_frame, height=10, width=70)
    self.result_text.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
    
    self.figure = Figure(figsize=(5, 4), dpi=100)
    self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
    self.canvas.get_tk_widget().pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)

def predict(self):
    iupac_or_smiles = self.entry.get()
    mol = Chem.MolFromSmiles(iupac_or_smiles)
    if mol is None:
        mol = Chem.MolFromIupac(iupac_or_smiles)
    
    if mol is None:
        messagebox.showerror("Error", "Invalid IUPAC name or SMILES.")
        return
    
    features = extract_features(Chem.MolToSmiles(mol))
    X_input = pd.DataFrame([features])
    
    gnn_pred = best_gnn_model(Data(x=torch.tensor(X_input.values[0], dtype=torch.float32)).to(device)).item()
    rf_pred = best_rf_model.predict(X_input)[0]
    
    gnn_pred_rescaled = scaler.inverse_transform([[gnn_pred]])[0][0]
    rf_pred_rescaled = scaler.inverse_transform([[rf_pred]])[0][0]
    
    gnn_r2 = r2_score(y_dat_test, best_gnn_model(Data(x=X_test_tensor, y=y_dat_test_tensor).to(device)).cpu().detach().numpy())
    rf_r2 = r2_score(y_dat_test, best_rf_model.predict(X_test))
    
    n = len(y_dat_test)
    p = X_test.shape[1]
    dof = n - p - 1
    t_value = stats.t.ppf(0.975, dof)
    
    gnn_pred_var = mean_squared_error(y_dat_test, best_gnn_model(Data(x=X_test_tensor, y=y_dat_test_tensor).to(device)).cpu().detach().numpy())
    rf_pred_var = mean_squared_error(y_dat_test, best_rf_model.predict(X_test))
    
    gnn_ci = t_value * np.sqrt(gnn_pred_var/n)
    rf_ci = t_value * np.sqrt(rf_pred_var/n)
    
    result_text = f"GNN Prediction:\n"
    result_text += f"Predicted pIC50: {gnn_pred_rescaled:.2f}\n"
    result_text += f"R-squared: {gnn_r2:.2f}\n"
    result_text += f"95% CI: [{gnn_pred_rescaled-gnn_ci:.2f}, {gnn_pred_rescaled+gnn_ci:.2f}]\n\n"
    
    result_text += f"Random Forest Prediction:\n"
    result_text += f"Predicted pIC50: {rf_pred_rescaled:.2f}\n"
    result_text += f"R-squared: {rf_r2:.2f}\n"
    result_text += f"95% CI: [{rf_pred_rescaled-rf_ci:.2f}, {rf_pred_rescaled+rf_ci:.2f}]"
    
    self.result_text.delete('1.0', tk.END)
    self.result_text.insert(tk.END, result_text)
    
    self.figure.clear()
    ax = self.figure.add_subplot(111)
    ax.bar(['GNN', 'Random Forest'], [gnn_pred_rescaled, rf_pred_rescaled])
    ax.set_ylabel("Predicted pIC50")
    ax.set_title("Model Comparison")
    self.canvas.draw()
if name == "main":
app = Application()
app.mainloop()