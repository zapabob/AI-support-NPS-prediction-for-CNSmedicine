import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import threading
import pandas as pd
import os
import dgl
import torch
import sys
import traceback

try:
    from utils.quantum_utils import extract_quantum_features
except ImportError:
    print("Error: Failed to import extract_quantum_features from utils.quantum_utils.")
    sys.exit(1)

class GUIApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Compound Activity Predictor")
        self.geometry("800x600")
        self.data_dir = os.path.join(os.getcwd(), 'data')
        self.create_widgets()
        self.prediction_thread = None
        self.prediction_event = threading.Event()

        try:
            self.load_data()
            self.load_models()
        except Exception as e:
            self.handle_error(e)

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

    def load_data(self):
        try:
            self.standard_compounds_data = pd.read_csv(os.path.join(self.data_dir, 'standard_compounds.csv'))
        except Exception as e:
            self.handle_error(e, "Error loading standard compounds data")

    def load_models(self):
        model_dir = os.path.join(self.data_dir, 'models')
        metrics_dir = os.path.join(self.data_dir, 'metrics')

        try:
            self.gcn_model = torch.load(os.path.join(model_dir, 'gcn_model.pt'))
            self.rf_model = pd.read_pickle(os.path.join(model_dir, 'rf_model.pkl'))
            self.models = {'gcn': self.gcn_model, 'rf': self.rf_model}
        except Exception as e:
            self.handle_error(e, "Error loading trained models")

        try:
            self.gcn_metrics = pd.read_csv(os.path.join(metrics_dir, 'gcn_metrics.csv'))
            self.rf_metrics = pd.read_csv(os.path.join(metrics_dir, 'rf_metrics.csv'))
        except Exception as e:
            self.handle_error(e, "Error loading evaluation metrics")

    def paste_input(self):
        try:
            iupac_or_smiles = self.clipboard_get()
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(tk.END, iupac_or_smiles)
        except tk.TclError as e:
            self.handle_error(e, "Error pasting input")

    def reset_input(self):
        self.input_entry.delete(0, tk.END)
        self.structure_canvas.delete("all")
        self.result_text.delete('1.0', tk.END)
        self.figure.clear()
        self.canvas.draw()

    def start_prediction(self):
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_event.clear()
            self.prediction_thread = threading.Thread(target=self.predict)
            self.prediction_thread.start()
            self.progress_bar.start()
            self.progress_label.config(text="Prediction in progress...")

    def stop_prediction(self):
        if self.prediction_thread is not None and self.prediction_thread.is_alive():
            self.prediction_event.set()
            self.prediction_thread.join()
        self.progress_bar.stop()
        self.progress_label.config(text="Prediction stopped.")
        self.time_label.config(text="")

    def predict(self):
        iupac_or_smiles = self.input_entry.get()
        try:
            mol = Chem.MolFromSmiles(iupac_or_smiles)
            if mol is None:
                mol = Chem.MolFromIupac(iupac_or_smiles)
        except Exception as e:
            self.handle_error(e, "Error parsing input")
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return

        if mol is None:
            self.progress_bar.stop()
            self.progress_label.config(text="")
            messagebox.showerror("Error", "Invalid IUPAC name or SMILES.")
            return

        try:
            img = Draw.MolToImage(mol, size=(300, 300))
            img = ImageTk.PhotoImage(img)
            self.structure_canvas.delete("all")
            self.structure_canvas.create_image(150, 150, image=img)
            self.structure_canvas.image = img
        except Exception as e:
            self.handle_error(e, "Error displaying 2D structure")
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return

        try:
            graph = dgl.smiles_to_bigraph(Chem.MolToSmiles(mol))
            descriptors = calculate_descriptors(mol)
            quantum_features = extract_quantum_features(mol)
            X_input = pd.DataFrame([pd.concat([graph, descriptors, quantum_features])])

            gcn_pred = self.models['gcn'](X_input.to(device)).item()
            rf_pred = self.models['rf'].predict(X_input)[0]

            gcn_pred_rescaled = self.scaler.inverse_transform([[gcn_pred]])[0][0]
            rf_pred_rescaled = self.scaler.inverse_transform([[rf_pred]])[0][0]
        except Exception as e:
            self.handle_error(e, "Error making predictions")
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return

        try:
            standard_predictions = {}
            for _, row in self.standard_compounds_data.iterrows():
                smiles = row['smiles']
                mol = Chem.MolFromSmiles(smiles)
                graph = dgl.smiles_to_bigraph(smiles)
                descriptors = calculate_descriptors(mol)
                quantum_features = extract_quantum_features(mol)
                X_input = pd.DataFrame([pd.concat([graph, descriptors, quantum_features])])
                gcn_pred = self.models['gcn'](X_input.to(device)).item()
                gcn_pred_rescaled = self.scaler.inverse_transform([[gcn_pred]])[0][0]
                standard_predictions[row['name']] = gcn_pred_rescaled
        except Exception as e:
            self.handle_error(e, "Error predicting for standard compounds")

        try:
            result_text = f"GCN Prediction:\nPredicted pIC50: {gcn_pred_rescaled:.2f}\n"
            result_text += f"Accuracy: {self.gcn_metrics['accuracy']:.2f}, F1-score: {self.gcn_metrics['f1_score']:.2f}\n"
            result_text += f"Confusion Matrix:\n{self.gcn_metrics['confusion_matrix']}\n\n"
            result_text += f"Random Forest Prediction:\nPredicted pIC50: {rf_pred_rescaled:.2f}\n"
            result_text += f"Accuracy: {self.rf_metrics['accuracy']:.2f}, F1-score: {self.rf_metrics['f1_score']:.2f}\n"
            result_text += f"Confusion Matrix:\n{self.rf_metrics['confusion_matrix']}\n\n"
            result_text += "Standard Compounds Predictions:\n"
            for name, pred in standard_predictions.items():
                result_text += f"{name}: {pred:.2f}\n"

            self.result_text.delete('1.0', tk.END)
            self.result_text.insert(tk.END, result_text)
        except Exception as e:
            self.handle_error(e, "Error displaying results")
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return

        try:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.bar(['GCN', 'Random Forest'], [gcn_pred_rescaled, rf_pred_rescaled])
            ax.set_ylabel("Predicted pIC50")
            ax.set_title("Model Comparison")
            self.canvas.draw()
        except Exception as e:
            self.handle_error(e, "Error displaying graph")
            self.progress_bar.stop()
            self.progress_label.config(text="")
            return

        self.progress_bar.stop()
        self.progress_label.config(text="Prediction completed.")
        self.time_label.config(text="")

    def handle_error(self, exception, error_message="An error occurred"):
        """
        Handle exceptions and display error messages.

        Args:
            exception (Exception): The exception object.
            error_message (str, optional): The error message to display.
        """
        print(f"{error_message}: {exception}")
        print(traceback.format_exc())
        messagebox.showerror("Error", f"{error_message}\n\n{exception}")

def calculate_descriptors(mol):
    """Calculate descriptors for a given compound"""
    descriptors = []

    # Basic descriptors
    descriptors.append(Descriptors.MolWt(mol))
    descriptors.append(Descriptors.HeavyAtomCount(mol))
    # Add other descriptors...

    return descriptors

def run_gui_app():
    app = GUIApp()
    app.mainloop()

if __name__ == "__main__":
    run_gui_app()
