import argparse
import numpy as np
from training.helper_functions import load_xtrain_ytrain
from xgboost import XGBRegressor


def parse_args():
    parser = argparse.ArgumentParser(description="Meta-model training for resampling strategy.")
    parser.add_argument(
        "--test-accuracies", type=float, nargs=6, required=True,
        metavar=('ENN', 'RU', 'RO', 'IS', 'SMOTE', 'TL'),
        help="Test accuracies for each resampling technique in order: ENN, RU, RO, IS, SMOTE, TL. Usage looks like:"
             "python meta_model_predictor.py --test-accuracies 0.85 0.78 0.82 0.80 0.79 0.81"
    )
    return parser.parse_args()


def extract_meta_features(X, y):
    f1 = np.sum(y == 0) / np.sum(y == 1)  # Imbalance Ratio
    f2 = X.shape[1]  # Number of features
    f3 = np.mean(np.abs(X[y == 1] - X[y == 0].mean(axis=0)))  # Overlap approximation
    return np.array([f1, f2, f3])


if __name__ == "__main__":
    args = parse_args()
    y_meta = np.array(args.test_accuracies)

    # Load resampled datasets
    print("[INFO] Extracting features...")
    X_train_enn, y_train_enn = load_xtrain_ytrain("data/train", "enn")
    X_train_ru, y_train_ru = load_xtrain_ytrain("data/train", "ru")
    X_train_ro, y_train_ro = load_xtrain_ytrain("data/train", "ro")
    X_train_is, y_train_is = load_xtrain_ytrain("data/train", "is")
    X_train_sm, y_train_sm = load_xtrain_ytrain("data/train", "smote")
    X_train_tl, y_train_tl = load_xtrain_ytrain("data/train", "tl")

    # Compute meta-features
    X_meta = np.array([
        extract_meta_features(X_train_enn, y_train_enn),
        extract_meta_features(X_train_ru, y_train_ru),
        extract_meta_features(X_train_ro, y_train_ro),
        extract_meta_features(X_train_is, y_train_is),
        extract_meta_features(X_train_sm, y_train_sm),
        extract_meta_features(X_train_tl, y_train_tl),
    ])

    # Train meta-model
    print("[INFO] Training meta-learner...")
    meta_model = XGBRegressor(n_estimators=100)
    meta_model.fit(X_meta, y_meta)

    # Predict on a new dataset
    X_new, y_new = load_xtrain_ytrain("data/train", "none")
    new_meta = extract_meta_features(X_new, y_new).reshape(1, -1)

    all_preds = meta_model.predict(X_meta)

    # Get the index of the best strategy
    best_idx = np.argmax(all_preds)

    # Map index to strategy name
    strategies = ["enn", "ru", "ro", "is", "smote", "tl"]
    best_strategy = strategies[best_idx]

    print(
        f"[INFO] Predicted best resampling strategy: {best_strategy.upper()} with estimated accuracy: {all_preds[best_idx]:.4f}")

