import optuna
from sklearn.model_selection import KFold, cross_val_score

def optimize_gcn_hyperparams(train_data, val_data, n_trials=100):
    """Optimize hyperparameters for GCN model using Optuna"""
    def objective(trial):
        hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
        n_epochs = trial.suggest_int('n_epochs', 50, 300)
        lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)

        model, _ = train_and_evaluate_gcn(train_data, val_data, hidden_dim=hidden_dim, n_epochs=n_epochs, lr=lr, dropout_rate=dropout_rate)
        val_loss = evaluate_gcn(model, val_data, device)

        return val_loss

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params

def optimize_rf_hyperparams(X, y, n_trials=100):
    """Optimize hyperparameters for Random Forest model using Optuna"""
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)
        max_depth = trial.suggest_int('max_depth', 2, 32)
        min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
        min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 4)

        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=42)
        scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        score = -scores.mean()

        return score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    return best_params
