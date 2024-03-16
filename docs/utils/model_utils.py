import torch
import torch.nn as nn
import torch.optim as optim
from dgllife.model import GCNPredictor
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import t, norm
import numpy as np
import optuna
from sklearn.linear_model import LassoCV
import traceback

# ROCm support
try:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
except Exception as e:
    print(f"Error setting up device: {e}")
    traceback.print_exc()

def train_gcn(model, optimizer, train_data, device, dropout_rate):
    """Train the GCN model"""
    try:
        model.train()
        total_loss = 0
        for graph, label in train_data:
            graph = graph.to(device)
            label = torch.tensor([label], dtype=torch.float32).to(device)
            optimizer.zero_grad()
            output = model(graph, dropout_rate=dropout_rate)
            loss = nn.MSELoss()(output, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_data)
    except Exception as e:
        print(f"Error training GCN model: {e}")
        traceback.print_exc()
        return float('inf')

def evaluate_gcn(model, data, device, dropout_rate=0.0):
    """Evaluate the GCN model"""
    try:
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for graph, label in data:
                graph = graph.to(device)
                label = torch.tensor([label], dtype=torch.float32).to(device)
                output = model(graph, dropout_rate=dropout_rate)
                loss = nn.MSELoss()(output, label)
                total_loss += loss.item()
        return total_loss / len(data)
    except Exception as e:
        print(f"Error evaluating GCN model: {e}")
        traceback.print_exc()
        return float('inf')

def train_and_evaluate_gcn(train_data, val_data, test_data, hidden_dim=64, n_epochs=100, lr=0.001, dropout_rate=0.2, l1_regularization=0.0):
    """Train and evaluate the GCN model"""
    try:
        model = GCNPredictor(in_feats=74, hidden_feats=[hidden_dim, hidden_dim], activation=[nn.ReLU(), nn.ReLU()], 
                             residual=[True, True], batchnorm=[True, True], dropout=[dropout_rate, dropout_rate]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l1_regularization)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        
        best_val_loss = float('inf')
        for epoch in range(n_epochs):
            train_loss = train_gcn(model, optimizer, train_data, device, dropout_rate)
            val_loss = evaluate_gcn(model, val_data, device, dropout_rate)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = model.state_dict()
            
            scheduler.step()
        
        model.load_state_dict(best_model)
        test_loss = evaluate_gcn(model, test_data, device, dropout_rate=0.0)
        return model, test_loss
    except Exception as e:
        print(f"Error training and evaluating GCN model: {e}")
        traceback.print_exc()
        return None, float('inf')

def train_and_evaluate_rf(X, y, n_splits=5, n_estimators=100, max_depth=None):
    """Train and evaluate the Random Forest model"""
    try:
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
        
        return model, np.mean(cv_scores)
    except Exception as e:
        print(f"Error training and evaluating Random Forest model: {e}")
        traceback.print_exc()
        return None, float('inf')

def calculate_confidence_interval(y_true, y_pred, alpha=0.05):
    """Calculate the coefficient of determination and confidence interval"""
    try:
        n = len(y_true)
        p = 1  # Number of predictor variables
        dof = n - p - 1  # Degrees of freedom
        t_value = t.ppf(1 - alpha / 2, dof)  # t-distribution value

        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        se = np.sqrt(mse / n)  # Standard error

        lower_bound = y_pred - t_value * se
        upper_bound = y_pred + t_value * se

        ci = norm.interval(1 - alpha, loc=y_pred, scale=se)

        return r2, ci
    except Exception as e:
        print(f"Error calculating confidence interval: {e}")
        traceback.print_exc()
        return 0.0, (0.0, 0.0)

def optimize_gcn_hyperparams(train_data, val_data, n_trials=100):
    """Optimize hyperparameters for GCN model using Optuna"""
    try:
        def objective(trial):
            hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
            n_epochs = trial.suggest_int('n_epochs', 50, 300)
            lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            l1_regularization = trial.suggest_float('l1_regularization', 0.0, 1.0)

            model, _ = train_and_evaluate_gcn(train_data, val_data, None, hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr, dropout_rate=dropout_rate, l1_regularization=l1_regularization)
            val_loss = evaluate_gcn(model, val_data, device, dropout_rate)

            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        return best_params
    except Exception as e:
        print(f"Error optimizing GCN hyperparameters: {e}")
        traceback.print_exc()
        return {}

def optimize_rf_hyperparams(X, y, n_trials=100):
    """Optimize hyperparameters for Random Forest model using Optuna"""
    try:
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 100, 1000)
            max_depth = trial.suggest_int('max_depth', 2, 32)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)
            alpha = trial.suggest_float('alpha', 0.0, 1.0)  # L1 regularization parameter

            model = LassoCV(alphas=[alpha], random_state=42, n_jobs=-1)
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            score = -scores.mean()

            return score

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        return best_params
    except Exception as e:
        print(f"Error optimizing Random Forest hyperparameters: {e}")
        traceback.print_exc()
        return {}
