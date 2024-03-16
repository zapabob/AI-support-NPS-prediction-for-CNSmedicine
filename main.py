import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_utils import load_chembl_data, normalize_data, prepare_data, split_data, standard_compounds
from model_utils import train_and_evaluate_gcn, train_and_evaluate_rf, optimize_gcn_hyperparams, optimize_rf_hyperparams, calculate_confidence_interval
from gui import run_gui_app
from chembl_webresource_client.utils import _get_data_folder, _setup_action_log_file
import traceback

def save_models_and_metrics(models, metrics, data_dir):
    """
    Save trained models and evaluation metrics to the /data directory.

    Args:
        models (dict): Dictionary containing the trained GCN and Random Forest models.
        metrics (dict): Dictionary containing the evaluation metrics for the GCN and Random Forest models.
        data_dir (str): Path to the /data directory.
    """
    try:
        model_dir = os.path.join(data_dir, 'models')
        metrics_dir = os.path.join(data_dir, 'metrics')

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)

        torch.save(models['gcn'].state_dict(), os.path.join(model_dir, 'gcn_model.pt'))
        pd.to_pickle(models['rf'], os.path.join(model_dir, 'rf_model.pkl'))

        pd.DataFrame(metrics['gcn']).to_csv(os.path.join(metrics_dir, 'gcn_metrics.csv'), index=False)
        pd.DataFrame(metrics['rf']).to_csv(os.path.join(metrics_dir, 'rf_metrics.csv'), index=False)
    except Exception as e:
        print(f"Error saving models and metrics: {e}")
        traceback.print_exc()

def main():
    """
    Main function to execute the compound activity prediction process.
    """
    try:
        # Set up the ChEMBL client
        chembl_data_folder = _get_data_folder()
        _setup_action_log_file()
    except Exception as e:
        print(f"Error setting up ChEMBL client: {e}")
        traceback.print_exc()
        return

    try:
        # Load data from ChEMBL
        target_ids = ['CHEMBL238', 'CHEMBL228', 'CHEMBL224', 'CHEMBL240', 'CHEMBL1833', 'CHEMBL1951', 'CHEMBL2039']
        chembl_data = load_chembl_data(target_ids)
    except Exception as e:
        print(f"Error loading ChEMBL data: {e}")
        traceback.print_exc()
        return

    try:
        # Preprocess data
        normalized_data, scaler = normalize_data(chembl_data)
        X, y = prepare_data(normalized_data)
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        traceback.print_exc()
        return

    try:
        # Split data into train, validation, and test sets
        train_data, val_data, test_data = split_data(X, y)
    except Exception as e:
        print(f"Error splitting data: {e}")
        traceback.print_exc()
        return

    try:
        # Optimize hyperparameters
        best_gcn_params = optimize_gcn_hyperparams(train_data, val_data)
        best_rf_params = optimize_rf_hyperparams(X, y)
    except Exception as e:
        print(f"Error optimizing hyperparameters: {e}")
        traceback.print_exc()
        return

    try:
        # Train and evaluate models with optimized hyperparameters
        gcn_model, gcn_metrics = train_and_evaluate_gcn(train_data, val_data, test_data, **best_gcn_params)
        rf_model, rf_metrics = train_and_evaluate_rf(X, y, **best_rf_params)
    except Exception as e:
        print(f"Error training and evaluating models: {e}")
        traceback.print_exc()
        return

    try:
        # Calculate confidence intervals and additional metrics
        y_true = scaler.inverse_transform(y.values.reshape(-1, 1)).flatten()
        X_test = X.iloc[test_data.index]
        y_gcn_pred = scaler.inverse_transform(gcn_model(X_test.to(device)).cpu().detach().numpy().reshape(-1, 1)).flatten()
        y_rf_pred = scaler.inverse_transform(rf_model.predict(X_test).reshape(-1, 1)).flatten()

        gcn_r2, gcn_ci = calculate_confidence_interval(y_true, y_gcn_pred)
        rf_r2, rf_ci = calculate_confidence_interval(y_true, y_rf_pred)

        gcn_metrics['r2'] = gcn_r2
        gcn_metrics['confidence_interval'] = gcn_ci
        rf_metrics['r2'] = rf_r2
        rf_metrics['confidence_interval'] = rf_ci
    except Exception as e:
        print(f"Error calculating confidence intervals and additional metrics: {e}")
        traceback.print_exc()

    models = {'gcn': gcn_model, 'rf': rf_model}
    metrics = {'gcn': gcn_metrics, 'rf': rf_metrics}

    # Save models and metrics to the /data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    save_models_and_metrics(models, metrics, data_dir)

    # Run the GUI application
    run_gui_app(models, scaler, standard_compounds)

if __name__ == "__main__":
    main()
