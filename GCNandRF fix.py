import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from chembl_webresource_client.new_client import new_client
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from io import BytesIO
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy import stats
import dgl
from dgllife.utils import smiles_to_bigraph, CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from dgllife.model import GCNPredictor

# Check if GPU is available
device = torch.device('GPU' if torch.cuda.is_available() else 'cpu')
# Load data from ChEMBL
def load_data():
    target_ids = ['CHEMBL238', 'CHEMBL228', 'CHEMBL1921', 'CHEMBL240']
    activities = new_client.activity
    res = activities.filter(target_chembl_id__in=target_ids).filter(standard_type="IC50")
    df = pd.DataFrame.from_records(res)
    df['pIC50'] = -np.log10(df['standard_value'] / 1e6)  # Convert to μM
    df = df[df['pIC50'] < 3.0]  # Exclude IC50 >= 1000μ
    return df, target_ids
# Normalize IC50 values across different experimental conditions
def normalize_ic50(df):
    scaler = StandardScaler()
    df['normalized_pIC50'] = scaler.fit_transform(df[['pIC50']])
    return df, scaler
 # Convert SMILES to DGLGraph with atom and bond features
def smiles_to_graph(smiles):
    graph = smiles_to_bigraph(smiles, node_featurizer=CanonicalAtomFeaturizer(), edge_featurizer=CanonicalBondFeaturizer())
    return graph
# Prepare data for training and evaluation
def prepare_data(df, scaler):
    X = df['canonical_smiles'].apply(smiles_to_graph)
    y_dat = df[df['target_chembl_id'] == 'CHEMBL238']['normalized_pIC50']
    y_net = df[df['target_chembl_id'] == 'CHEMBL228']['normalized_pIC50']
    y_5ht2a = df[df['target_chembl_id'] == 'CHEMBL1921']['normalized_pIC50']
    y_herg = df[df['target_chembl_id'] == 'CHEMBL240']['normalized_pIC50']
    X_train, X_test, y_dat_train, y_dat_test, y_net_train, y_net_test, y_5ht2a_train, y_5ht2a_test, y_herg_train, y_herg_test = train_test_split(
        X, y_dat, y_net, y_5ht2a, y_herg, test_size=0.2, random_state=42)
    X_train, X_val, y_dat_train, y_dat_val, y_net_train, y_net_val, y_5ht2a_train, y_5ht2a_val, y_herg_train, y_herg_val = train_test_split(
        X_train, y_dat_train, y_net_train, y_5ht2a_train, y_herg_train, test_size=0.2, random_state=42)
    train_data = list(zip(X_train, y_dat_train))
    val_data = list(zip(X_val, y_dat_val))
    test_data = list(zip(X_test, y_dat_test))
    return train_data, val_data, test_data, scaler
# Train GCN model
def train_gcn(model, optimizer, train_data, device):
    model.train()
    total_loss = 0
    for graph, label in train_data:
        graph = graph.to(device)
        label = torch.tensor([label], dtype=torch.float32).to(device)
        optimizer.zero_grad()
        output = model(graph)
        loss = nn.MSELoss()(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_data)
# Evaluate GCN model
def evaluate_gcn(model, data, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for graph, label in data:
            graph = graph.to(device)
            label = torch.tensor([label], dtype=torch.float32).to(device)
            output = model(graph)
            loss = nn.MSELoss()(output, label)
            total_loss += loss.item()
    return total_loss / len(data)
# Train and evaluate GCN model with cross-validation
def train_and_evaluate_gcn(train_data, val_data, test_data, hidden_dim=64, n_epochs=100, lr=0.001):
    model = GCNPredictor(in_feats=74, hidden_feats=[hidden_dim, hidden_dim], activation=[nn.ReLU(), nn.ReLU()], 
                        residual=[True, True], batchnorm=[True, True], dropout=[0.2, 0.2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    best_val_loss = float('inf')
    for epoch in range(n_epochs):
        train_loss = train_gcn(model, optimizer, train_data, device)
        val_loss = evaluate_gcn(model, val_data, device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model.state_dict()
        scheduler.step()
    model.load_state_dict(best_model)
    test_loss = evaluate_gcn(model, test_data, device)
    return model, test_loss
# Train and evaluate Random Forest model with cross-validation
def train_and_evaluate_rf(X, y, n_splits=5, n_estimators=100, max_depth=None):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        cv_scores.append(mse)
    return np.mean(cv_scores)
# Create GUI
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compound Activity Predictor")
        self.geometry("800x600")
        self.create_widgets()
    def create_widgets(self):
        self.input_frame = ttk.LabelFrame(self, text="Input")
        self.input_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.input_label = ttk.Label(self.input_frame, text="Enter IUPAC name or SMILES:")
        self.input_label.pack(pady=5)
        self.input_entry = ttk.Entry(self.input_frame, width=50)
        self.input_entry.pack()
        self.input_buttons_frame = ttk.Frame(self.input_frame)
        self.input_buttons_frame.pack(pady=5)
        self.paste_button = ttk.Button(self.input_buttons_frame, text="Paste", command=self.paste_input)
        self.paste_button.pack(side=tk.LEFT, padx=5)
        self.reset_button = ttk.Button(self.input_buttons_frame, text="Reset", command=self.reset_input)
        self.reset_button.pack(side=tk.LEFT, padx=5)
        self.predict_button = ttk.Button(self.input_frame, text="Predict", command=self.predict)
        self.predict_button.pack(pady=10)
        self.stop_button = ttk.Button(self.input_frame, text="Stop", command=self.stop_prediction)
        self.stop_button.pack()
        self.structure_frame = ttk.LabelFrame(self, text="2D Structure")
        self.structure_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.structure_canvas = tk.Canvas(self.structure_frame, width=300, height=300)
        self.structure_canvas.pack()
        self.result_frame = ttk.LabelFrame(self, text="Prediction Results")
        self.result_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.result_text = tk.Text(self.result_frame, height=10, width=70)
        self.result_text.pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(5, 4), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.result_frame)
        self.canvas.get_tk_widget().pack(side=tk.LEFT, padx=5, pady=5, fill=tk.BOTH, expand=True)
        self.progress_frame = ttk.LabelFrame(self, text="Progress")
        self.progress_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        self.progress_label = ttk.Label(self.progress_frame, text="Prediction in progress...")
        self.progress_label.pack()
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=200, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        self.time_label = ttk.Label(self.progress_frame, text="Estimated time remaining: ")
        self.time_label.pack()
    def paste_input(self):
        try:
            iupac_or_smiles = self.clipboard_get()
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(tk.END, iupac_or_smiles)
        except tk.TclError:
            messagebox.showerror("Error", "Failed to paste from clipboard.")
    def reset_input(self):
        self.input_entry.delete(0, tk.END)
        self.structure_canvas.delete("all")
        self.result_text.delete('1.0', tk.END)
        self.figure.clear()
        self.canvas.draw()
    def stop_prediction(self):
        self.progress_bar.stop()
        self.progress_label.config(text="Prediction stopped.")
        self.time_label.config(text="")
    def predict(self):
        iupac_or_smiles = self.input_entry.get()
        mol = Chem.MolFromSmiles(iupac_or_smiles)
        if mol is None:
            mol = Chem.MolFromIupac(iupac_or_smiles)
        if mol is None:
            messagebox.showerror("Error", "Invalid IUPAC name or SMILES.")
            return
        self.progress_bar.start()
        self.progress_label.config(text="Prediction in progress...")
        # Display 2D structure
        img = Draw.MolToImage(mol, size=(300, 300))
        img = ImageTk.PhotoImage(img)
        self.structure_canvas.delete("all")
        self.structure_canvas.create_image(150, 150, image=img)
        self.structure_canvas.image = img
        # Make predictions using the trained models
        graph = smiles_to_graph(Chem.MolToSmiles(mol))
        graph = graph.to(device)
        gcn_model, gcn_test_loss = train_and_evaluate_gcn(train_data, val_data, test_data, hidden_dim=64, n_epochs=100, lr=0.001)
        gcn_pred = gcn_model(graph).item()
        X_input = pd.DataFrame([mol])
        rf_pred = rf_model.predict(X_input)[0]
        gcn_pred_rescaled = scaler.inverse_transform([[gcn_pred]])[0][0]
        rf_pred_rescaled = scaler.inverse_transform([[rf_pred]])[0][0]
        n = len([label for _, label in test_data])
        p = 1
        dof = n - p - 1
        t_value = stats.t.ppf(0.975, dof)
        gcn_pred_var = mean_squared_error([label for _, label in test_data], gcn_model(graph.to(device)).cpu().detach().numpy())
        rf_pred_var = mean_squared_error([label for _, label in test_data], rf_model.predict(X_test))
        gcn_ci = t_value * np.sqrt(gcn_pred_var/n)
        rf_ci = t_value * np.sqrt(rf_pred_var/n)
        result_text = f"GCN Prediction:\n"
        result_text += f"Predicted pIC50: {gcn_pred_rescaled:.2f}\n"
        result_text += f"95% CI: [{gcn_pred_rescaled-gcn_ci:.2f}, {gcn_pred_rescaled+gcn_ci:.2f}]\n\n"
        
        result_text += f"Random Forest Prediction:\n"
        result_text += f"Predicted pIC50: {rf_pred_rescaled:.2f}\n"
        result_text += f"95% CI: [{rf_pred_rescaled-rf_ci:.2f}, {rf_pred_rescaled+rf_ci:.2f}]"
        
        self.result_text.delete('1.0', tk.END)
        self.result_text.insert(tk.END, result_text)
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(['GCN', 'Random Forest'], [gcn_pred_rescaled, rf_pred_rescaled])
        ax.set_ylabel("Predicted pIC50")
        ax.set_title("Model Comparison")
        self.canvas.draw()
        self.progress_bar.stop()
        self.progress_label.config(text="Prediction completed.")
        self.time_label.config(text="")

if __name__ == "__main__":
    df, target_ids = load_data()
    df, scaler = normalize_ic50(df)
    train_data, val_data, test_data, scaler = prepare_data(df, scaler)

    X_train = pd.DataFrame([graph for graph, _ in train_data])
    y_train = pd.Series([label for _, label in train_data])
    X_test = pd.DataFrame([graph for graph, _ in test_data])
    y_test = pd.Series([label for _, label in test_data])

    rf_model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_test_loss = mean_squared_error(y_test, rf_model.predict(X_test))

    gcn_model, gcn_test_loss = train_and_evaluate_gcn(train_data, val_data, test_data, hidden_dim=64, n_epochs=100, lr=0.001)

    print(f"GCN Test Loss: {gcn_test_loss:.4f}")
    print(f"Random Forest Test Loss: {rf_test_loss:.4f}")

    app = Application()
    app.mainloop()
