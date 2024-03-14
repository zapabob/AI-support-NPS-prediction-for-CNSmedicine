import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Draw
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from utils.data_utils import smiles_to_graph, standard_compounds
import threading
import pandas as pd

class GUIApp(tk.Tk):
    def __init__(self, models, scaler):
        super().__init__()
        self.title("Compound Activity Predictor")
        self.geometry("800x600")
        self.models = models
        self.scaler = scaler
        self.create_widgets()
        self.prediction_thread = None

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
        
        self.predict_button = ttk.Button(self.input_frame, text="Predict", command=self.start_prediction)
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
        
        self.progress_label = ttk.Label(self.progress_frame, text="")
        self.progress_label.pack()
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, length=200, mode='indeterminate')
        self.progress_bar.pack(pady=5)
        
        self.time_label = ttk.Label(self.progress_frame, text="")
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

    def start_prediction(self):
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_thread = threading.Thread(target=self.predict)
            self.prediction_thread.start()
            self.progress_bar.start()
            self.progress_label.config(text="Prediction in progress...")

    def stop_prediction(self):
        if self.prediction_thread is not None and self.prediction_thread.is_alive():
            self.prediction_thread.join()
        self.progress_bar.stop()
        self.progress_label.config(text="Prediction stopped.")
        self.time_label.config(text="")

    def predict(self):
        iupac_or_smiles = self.input_entry.get()
        mol = Chem.MolFromSmiles(iupac_or_smiles)
        if mol is None:
            mol = Chem.MolFromIupac(iupac_or_smiles)
        
        if mol is None:
            self.progress_bar.stop()
            self.progress_label.config(text="")
            messagebox.showerror("Error", "Invalid IUPAC name or SMILES.")
            return
        
        # Display 2D structure
        img = Draw.MolToImage(mol, size=(300, 300))
        img = ImageTk.PhotoImage(img)
        self.structure_canvas.delete("all")
        self.structure_canvas.create_image(150, 150, image=img)
        self.structure_canvas.image = img
        
        # Make predictions using the trained models
        graph = smiles_to_graph(Chem.MolToSmiles(mol))
        graph = graph.to(device)
        gcn_pred = self.models['gcn'](graph).item()
        
        X_input = pd.DataFrame([mol])
        rf_pred = self.models['rf'].predict(X_input)[0]
        
        gcn_pred_rescaled = self.scaler.inverse_transform([[gcn_pred]])[0][0]
        rf_pred_rescaled = self.scaler.inverse_transform([[rf_pred]])[0][0]
        
        # Predict NPS for standard compounds
        standard_predictions = {}
        for name, smiles in standard_compounds.items():
            mol = Chem.MolFromSmiles(smiles)
            graph = smiles_to_graph(smiles)
            graph = graph.to(device)
            gcn_pred = self.models['gcn'](graph).item()
            gcn_pred_rescaled = self.scaler.inverse_transform([[gcn_pred]])[0][0]
            standard_predictions[name] = gcn_pred_rescaled
        
        # Calculate coefficient of determination and confidence interval
        y_true = self.scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
        y_gcn_pred = self.scaler.inverse_transform(gcn_model(X_test.to(device)).cpu().detach().numpy().reshape(-1, 1)).flatten()
        y_rf_pred = self.scaler.inverse_transform(rf_model.predict(X_test).reshape(-1, 1)).flatten()
        
        gcn_r2, gcn_ci = calculate_confidence_interval(y_true, y_gcn_pred)
        rf_r2, rf_ci = calculate_confidence_interval(y_true, y_rf_pred)
        
        result_text = f"GCN Prediction:\nPredicted pIC50: {gcn_pred_rescaled:.2f}\n"
        result_text += f"R^2: {gcn_r2:.2f}, 95% CI: [{gcn_ci[0]:.2f}, {gcn_ci[1]:.2f}]\n\n"
        result_text += f"Random Forest Prediction:\nPredicted pIC50: {rf_pred_rescaled:.2f}\n"
        result_text += f"R^2: {rf_r2:.2f}, 95% CI: [{rf_ci[0]:.2f}, {rf_ci[1]:.2f}]\n\n"
        result_text += "Standard Compounds Predictions:\n"
        for name, pred in standard_predictions.items():
            result_text += f"{name}: {pred:.2f}\n"
        
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

def run_gui_app(models, scaler):
    app = GUIApp(models, scaler)
    app.mainloop()
