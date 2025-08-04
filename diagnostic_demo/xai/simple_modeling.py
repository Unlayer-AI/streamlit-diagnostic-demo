from sklearn.linear_model import LogisticRegression
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score
)
import pandas as pd

def run_simple_modeling(X, y, model) -> pd.DataFrame:
    result = {
        "original_model": {},
        "lr": {},
        "ebm": {},
    }

    # Calculate metrics for the original model
    preds = model.predict(X)
    result["original_model"]["accuracy"] = accuracy_score(y, preds)
    result["original_model"]["balanced_accuracy"] = balanced_accuracy_score(y, preds)
    result["original_model"]["f1"] = f1_score(y, preds)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=100)
    log_reg_param_grid = {
        "C": [0.01, 1, 100],
        "solver": ["saga", "lbfgs"],
    }

    ebm = ExplainableBoostingClassifier(
        interactions=5,
    )
    ebm_param_grid = {
        "smoothing_rounds": [0, 100],
        "learning_rate": [0.005, 0.05],
        "interactions": [5, 20],
        "interaction_smoothing_rounds": [0, 100],
        "min_samples_leaf": [2, 8],
    }

    skf = StratifiedKFold(n_splits=3)
    for name, model, params in [
        ("lr", log_reg, log_reg_param_grid),
        ("ebm", ebm, ebm_param_grid),
    ]:
        if name == "lr":
            # LR needs scaling (in case it was not done)
            scaler = StandardScaler()
            for col in X.columns:
                # if it's numerical, scale it
                if X[col].dtype in ["int64", "float64"]:
                    X[col] = scaler.fit_transform(X[[col]])
            # tuning
            grid_search = RandomizedSearchCV(
                model,
                params,
                cv=skf,
                scoring="accuracy",
                n_iter=3
            )
            grid_search.fit(X, y)
            best_model = grid_search.best_estimator_
        else:
            # too expensive to tune, so we use default parameters
            best_model = ebm

        # Validate the best model on each fold and average the scores
        accuracy_scores = []
        balanced_accuracy_scores = []
        f1_scores = []

        for idx_split, (train_index, val_index) in enumerate(skf.split(X, y)):
            print(f"Fitting {name} on split no {idx_split}")
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            best_model.fit(X_train, y_train)
            val_preds = best_model.predict(X_val)

            accuracy_scores.append(accuracy_score(y_val, val_preds))
            balanced_accuracy_scores.append(balanced_accuracy_score(y_val, val_preds))
            f1_scores.append(f1_score(y_val, val_preds))

        result[name]["accuracy"] = sum(accuracy_scores) / len(accuracy_scores)
        result[name]["balanced_accuracy"] = sum(balanced_accuracy_scores) / len(
            balanced_accuracy_scores
        )
        result[name]["f1"] = sum(f1_scores) / len(f1_scores)

    # Compare performances
    all_results = {
        "metric": ["accuracy", "balanced accuracy", "F1 score"],
        "original model": [
            result["original_model"]["accuracy"],
            result["original_model"]["balanced_accuracy"],
            result["original_model"]["f1"],
        ],
        "logistic regression (est.)": [
            result["lr"]["accuracy"],
            result["lr"]["balanced_accuracy"],
            result["lr"]["f1"],
        ],
        "explainable boosting (est.)": [
            result["ebm"]["accuracy"],
            result["ebm"]["balanced_accuracy"],
            result["ebm"]["f1"],
        ],
    }
    
    comparison_df = pd.DataFrame(all_results).set_index("metric", drop=True)
    
    return comparison_df