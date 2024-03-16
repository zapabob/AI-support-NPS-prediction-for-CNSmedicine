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
    from utils.descriptor_utils import calculate_descriptors
except ImportError:
    print("Error: Failed to import required modules from utils.")
    sys.exit(1)

class GUIApp(tk.Tk):
    def __init__(self, models, scaler, standard_compounds):
        super().__init__()
        self.title("Compound Activity Predictor")
        self.geometry("800x600")
        self.models = models
        self.scaler = scaler
        self.standard_compounds = standard_compounds
        self.create_widgets()
        self.prediction_thread = None
        self.prediction_event = threading.Event()

    def create_widgets(self):
        """Create the GUI widgets."""
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
        """Paste the input from the clipboard."""
        try:
            iupac_or_smiles = self.clipboard_get()
            self.input_entry.delete(0, tk.END)
            self.input_entry.insert(tk.END, iupac_or_smiles)
        except tk.TclError as e:
            self.handle_error(e, "Error pasting input")

    def reset_input(self):
        """Reset the input field and clear the results."""
        self.input_entry.delete(0, tk.END)
        self.structure_canvas.delete("all")
        self.result_text.delete('1.0', tk.END)
        self.figure.clear()
        self.canvas.draw()

    def start_prediction(self):
        """Start the prediction process in a separate thread."""
        if self.prediction_thread is None or not self.prediction_thread.is_alive():
            self.prediction_event.clear()
            self.prediction_thread = threading.Thread(target=self.predict)
            self.prediction_thread.start()
            self.progress_bar.start()
            self.progress_label.config(text="Prediction in progress...")

    def stop_prediction(self):
        """Stop the prediction process."""
        if self.prediction_thread is not None and self.prediction_thread.is_alive():
            self.prediction_event.set()
            self.prediction_thread.join()
        self.progress_bar.stop()
        self.progress_label.config(text="Prediction stopped.")
        self.time_label.config(text="")

    def predict(self):
        """Perform the prediction and display the results."""
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
            for name, smiles in self.standard_compounds.items():
                mol = Chem.MolFromSmiles(smiles)
                graph = dgl.smiles_to_bigraph(smiles)
                descriptors = calculate_descriptors(mol)
                quantum_features = extract_quantum_features(mol)
                X_input = pd.DataFrame([pd.concat([graph, descriptors, quantum_features])])
                gcn_pred = self.models['gcn'](X_input.to(device)).item()
                gcn_pred_rescaled = self.scaler.inverse_transform([[gcn_pred]])[0][0]
                standard_predictions[name] = gcn_pred_rescaled
        except Exception as e:
            self.handle_error(e, "Error predicting for standard compounds")

        try:
            result_text = f"GCN Prediction:\nPredicted pIC50: {gcn_pred_rescaled:.2f}\n"
            result_text += f"Accuracy: {self.models['gcn'].metrics['accuracy']:.2f}, F1-score: {self.models['gcn'].metrics['f1_score']:.2f}\n"
            result_text += f"Confusion Matrix:\n{self.models['gcn'].metrics['confusion_matrix']}\n"
            result_text += f"R^2: {self.models['gcn'].metrics['r2']:.2f}, 95% CI: [{self.models['gcn'].metrics['confidence_interval'][0]:.2f}, {self.models['gcn'].metrics['confidence_interval'][1]:.2f}]\n\n"
            result_text += f"Random Forest Prediction:\nPredicted pIC50: {rf_pred_rescaled:.2f}\n"
            result_text += f"Accuracy: {self.models['rf'].metrics['accuracy']:.2f}, F1-score: {self.models['rf'].metrics['f1_score']:.2f}\n"
            result_text += f"Confusion Matrix:\n{self.models['rf'].metrics['confusion_matrix']}\n"
            result_text += f"R^2: {self.models['rf'].metrics['r2']:.2f}, 95% CI: [{self.models['rf'].metrics['confidence_interval'][0]:.2f}, {self.models['rf'].metrics['confidence_interval'][1]:.2f}]\n\n"
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

def run_gui_app(models, scaler, standard_compounds):
    """
    Run the GUI application.

    Args:
        models (dict): Dictionary containing the trained GCN and Random Forest models.
        scaler (object): Scaler object used for data normalization.
        standard_compounds (dict): Dictionary containing the SMILES of standard compounds.
    """
    app = GUIApp(models, scaler, standard_compounds)
    app.mainloop()
