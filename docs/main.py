import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_utils import load_data, normalize_data, prepare_data
from model_utils import train_and_evaluate_models, optimize_gcn_hyperparams, optimize_rf_hyperparams
from gui import run_gui_app

def save_models_and_metrics(models, metrics, data_dir):
    """Save trained models and evaluation metrics to /data directory"""
    model_dir = os.path.join(data_dir, 'models')
    metrics_dir = os.path.join(data_dir, 'metrics')

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    torch.save(models['gcn'].state_dict(), os.path.join(model_dir, 'gcn_model.pt'))
    pd.to_pickle(models['rf'], os.path.join(model_dir, 'rf_model.pkl'))

    pd.DataFrame(metrics['gcn']).to_csv(os.path.join(metrics_dir, 'gcn_metrics.csv'), index=False)
    pd.DataFrame(metrics['rf']).to_csv(os.path.join(metrics_dir, 'rf_metrics.csv'), index=False)

def main():
    # Load data from ChEMBL
    target_ids = ['CHEMBL238', 'CHEMBL228', 'CHEMBL1921', 'CHEMBL240', 'CHEMBL3945', 'CHEMBL4214', 'CHEMBL4497']
    chembl_data = load_chembl_data(target_ids)

    # Preprocess data
    X, y, scaler = preprocess_data(chembl_data)

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(X, y)

    # Optimize hyperparameters
    best_gcn_params = optimize_gcn_hyperparams(train_data, val_data)
    best_rf_params = optimize_rf_hyperparams(X, y)

    # Train and evaluate models with optimized hyperparameters
    models, metrics = train_and_evaluate_models(train_data, val_data, test_data, best_gcn_params, best_rf_params)

    # Save models and metrics to /data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    save_models_and_metrics(models, metrics, data_dir)

    # Run the GUI application
    run_gui_app(models, scaler)

if __name__ == "__main__":
    main()
