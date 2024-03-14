import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_utils import load_data, normalize_data, prepare_data
from model_utils import train_and_evaluate_models
from gui import run_gui_app

def main():
    # Load and preprocess data
    data = load_data()
    normalized_data, scaler = normalize_data(data)
    train_data, val_data, test_data = prepare_data(normalized_data)

    # Train and evaluate models
    models, metrics = train_and_evaluate_models(train_data, val_data, test_data)

    # Run the GUI application
    run_gui_app(models, scaler)

if __name__ == "__main__":
    main()
