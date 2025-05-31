# Import Libraries
import pickle
import joblib
import mlflow
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from lime.lime_tabular import LimeTabularExplainer

# Load model form MLflow
mlruns_uri = Path("../mlruns").resolve().as_uri()
mlflow.set_tracking_uri(mlruns_uri)
model_name = "Final XGB + SMOTE Tuned Model"
prod_v = (
    mlflow.MlflowClient()
    .get_latest_versions(model_name, stages=["Production"])[0]
    .version
)
model = mlflow.sklearn.load_model(f"models:/{model_name}/{prod_v}")

# Load data
X_train, X_test, y_train, y_test = joblib.load(
    "../data/processed/split_data.pkl"
)
proba_test = model.predict_proba(X_test)[:, 1]

# Define slider grids
avg_fraud_range = list(range(100, 141, 20))
tp_fee_range = list(range(5, 16, 5))
fp_cost_range = [round(x, 2) for x in np.arange(2.00, 5.01, 1.00)]
fn_penalty_range = list(range(0, 51, 25))

# Helper functions


def profit_score(
    y_true, y_pred, avg_fraud, tp_fee_pct, fp_cost, fn_penalty_pct
):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp*tp_fee_pct*avg_fraud - fp*fp_cost - fn*fn_penalty_pct*avg_fraud


def best_threshold(
    y_true, proba, avg_fraud, tp_fee_pct, fp_cost, fn_penalty_pct
):
    taus = np.linspace(0, 1, 401)
    profits = [
        profit_score(
            y_true, proba >= t, avg_fraud, tp_fee_pct, fp_cost, fn_penalty_pct
        )
        for t in taus
    ]
    best_i = int(np.argmax(profits))
    return taus[best_i], profits[best_i]


# LIME explainer
explainer = LimeTabularExplainer(
    training_data=X_test.values,
    feature_names=X_test.columns.tolist(),
    class_names=["Legit", "Fraud"],
    mode="classification",
    discretize_continuous=True,
)

# Iterate & cache results
pen_grid = np.linspace(0, 1, 101)
profit_cache = {}
profit_cache["pen_grid"] = pen_grid.tolist()
total = (
    len(avg_fraud_range)
    * len(tp_fee_range)
    * len(fp_cost_range)
    * len(fn_penalty_range)
)
counter = 1

print("Pre-computing ...")
for avg_fraud in avg_fraud_range:
    for tp_fee in tp_fee_range:
        for fp_cost in fp_cost_range:
            for fn_penalty in fn_penalty_range:
                print(f" Processing {counter}/{total}", end="\r")
                key = (avg_fraud, tp_fee, fp_cost, fn_penalty)

                tp_fee_pct = tp_fee / 100
                fn_penalty_pct = fn_penalty / 100

                # Profits & threshold
                tau_opt, profit_opt = best_threshold(
                    y_test,
                    proba_test,
                    avg_fraud,
                    tp_fee_pct,
                    fp_cost,
                    fn_penalty_pct
                )
                profit_fixed = profit_score(
                    y_test,
                    proba_test >= 0.5,
                    avg_fraud,
                    tp_fee_pct,
                    fp_cost,
                    fn_penalty_pct
                )

                # Confusion matrices
                pred_opt = proba_test >= tau_opt
                pred_fixed = proba_test >= 0.5
                cm_opt = confusion_matrix(y_test, pred_opt).ravel().tolist()
                cm_fixed = confusion_matrix(
                    y_test, pred_fixed
                ).ravel().tolist()

                # Profit curves across penalty grid
                curve_best, curve_fixed = [], []
                for p in pen_grid:
                    tau_tmp, _ = best_threshold(
                        y_test, proba_test, avg_fraud, tp_fee_pct, fp_cost, p
                    )
                    curve_best.append(
                        profit_score(
                            y_test,
                            proba_test >= tau_tmp,
                            avg_fraud,
                            tp_fee_pct,
                            fp_cost,
                            p
                        )
                    )
                    curve_fixed.append(
                        profit_score(
                            y_test,
                            pred_fixed,
                            avg_fraud,
                            tp_fee_pct,
                            fp_cost,
                            p
                        )
                    )

                # 2 examples of each outcome for LIME
                tp_idx = (
                    np.where((y_test == 1) & (pred_opt == 1))[0][:2].tolist()
                )
                tn_idx = np.where(
                    (y_test == 0) & (pred_opt == 0)
                )[0][:2].tolist()
                fp_idx = np.where(
                    (y_test == 0) & (pred_opt == 1)
                )[0][:2].tolist()
                fn_idx = np.where(
                    (y_test == 1) & (pred_opt == 0)
                )[0][:2].tolist()
                example_indices = {
                    "TP": tp_idx,
                    "TN": tn_idx,
                    "FP": fp_idx,
                    "FN": fn_idx,
                }

                lime_explanations = {}
                for cat, idx_list in example_indices.items():
                    lime_explanations[cat] = []
                    for i in idx_list:
                        exp = explainer.explain_instance(
                            X_test.values[i],
                            model.predict_proba,
                            num_features=8
                        )
                        lime_explanations[cat].append(exp.as_list())

                # Store values in cache
                profit_cache[key] = {
                    "tau_opt": tau_opt,
                    "profit_opt": profit_opt,
                    "profit_fixed": profit_fixed,
                    "cm_opt": cm_opt,
                    "cm_fixed": cm_fixed,
                    "curve_best": curve_best,
                    "curve_fixed": curve_fixed,
                    "example_indices": example_indices,
                    "lime_explanations": lime_explanations,
                }

                counter += 1

# Save the cache to pickle file
print("\nSaving cache ...")
Path("../data/processed").mkdir(parents=True, exist_ok=True)
with open("../data/processed/profit_cache.pkl", "wb") as f:
    pickle.dump(profit_cache, f)
print("Done - cached file written to ../data/processed/profit_cache.pkl")
