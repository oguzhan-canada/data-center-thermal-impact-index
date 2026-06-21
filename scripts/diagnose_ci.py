"""
Diagnose why CI coverage is 26.9% instead of 85%.
Checks raw quantile interval widths and residual distribution.
"""
import logging
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

from pipeline.predict_train import (
    build_training_matrix,
    engineer_features,
    get_splits,
    tune_hyperparameters,
    train_final_model,
    train_stratified_day_models,
    FEATURE_COLUMNS,
    TARGET_NIGHT,
    TEST_REGION,
)


def diagnose_ci_coverage(model_med, model_q10, model_q90,
                          cal_df, test_df, target_col, label):
    X_cal  = cal_df[FEATURE_COLUMNS].values
    y_cal  = cal_df[target_col].values
    X_test = test_df[FEATURE_COLUMNS].values
    y_test = test_df[target_col].values

    # Raw interval widths before any correction
    cal_q10  = model_q10.predict(X_cal)
    cal_q90  = model_q90.predict(X_cal)
    test_q10 = model_q10.predict(X_test)
    test_q90 = model_q90.predict(X_test)

    print(f"\n{'='*60}")
    print(f"CI DIAGNOSIS: {label}")
    print(f"  Calibration set size: {len(y_cal)} rows")
    print(f"  Test set size:        {len(y_test)} rows")
    print(f"\n  RAW QUANTILE INTERVAL WIDTHS (before correction):")
    print(f"  Cal  q90-q10: mean={np.mean(cal_q90-cal_q10):.4f}  "
          f"std={np.std(cal_q90-cal_q10):.4f}  "
          f"min={np.min(cal_q90-cal_q10):.4f}")
    print(f"  Test q90-q10: mean={np.mean(test_q90-test_q10):.4f}  "
          f"std={np.std(test_q90-test_q10):.4f}  "
          f"min={np.min(test_q90-test_q10):.4f}")
    print(f"\n  ACTUAL LABEL SPREAD:")
    print(f"  Cal  y spread: std={np.std(y_cal):.4f}  "
          f"range=[{np.min(y_cal):.3f}, {np.max(y_cal):.3f}]")
    print(f"  Test y spread: std={np.std(y_test):.4f}  "
          f"range=[{np.min(y_test):.3f}, {np.max(y_test):.3f}]")
    print(f"\n  RAW COVERAGE (no correction):")
    raw_cov = np.mean((y_test >= test_q10) & (y_test <= test_q90))
    print(f"  {raw_cov:.1%}")
    print(f"\n  NEEDED CORRECTION (to hit 85%):")
    # How wide do intervals need to be?
    pred_med = model_med.predict(X_test)
    needed_half_width = np.percentile(np.abs(pred_med - y_test), 85)
    print(f"  Correction needed ~ {needed_half_width:.4f} deg C "
          f"(based on residual distribution)")

    # Also show what conformal would compute on cal set
    nonconformity = np.maximum(cal_q10 - y_cal, y_cal - cal_q90)
    nonconformity = np.maximum(nonconformity, 0)
    n = len(y_cal)
    level = np.ceil((n + 1) * 0.90) / n
    level = min(level, 1.0)
    correction = float(np.quantile(nonconformity, level))
    print(f"\n  CONFORMAL CORRECTION (from cal set):")
    print(f"  correction = {correction:.4f}°C at level {level:.4f}")
    corrected_cov = np.mean(
        (y_test >= test_q10 - correction) & (y_test <= test_q90 + correction)
    )
    print(f"  Corrected test coverage = {corrected_cov:.1%}")
    print(f"{'='*60}\n")


def main():
    print("Building training matrix...")
    df_raw = build_training_matrix()
    df = engineer_features(df_raw)
    train_val, test, loro_folds = get_splits(df)
    print(f"Train+val: {len(train_val)} | Test ({TEST_REGION}): {len(test)}")

    # Quick train night model (10 trials for speed)
    print("\nTraining night model (10 trials for diagnosis)...")
    best_params = tune_hyperparameters(
        train_val, loro_folds, TARGET_NIGHT, n_trials=10, seed=42,
    )
    models_night = train_final_model(train_val, best_params, TARGET_NIGHT, seed=42)

    # Use 15% of train_val as cal set (same as production code)
    cal_night = train_val.sample(frac=0.15, random_state=42)

    diagnose_ci_coverage(
        models_night[0], models_night[1], models_night[2],
        cal_night, test, TARGET_NIGHT, "NIGHT"
    )

    # Also try with full train_val as cal set
    diagnose_ci_coverage(
        models_night[0], models_night[1], models_night[2],
        train_val, test, TARGET_NIGHT, "NIGHT (full train_val as cal)"
    )

    # Day CEM
    print("\nTraining day CEM model (10 trials for diagnosis)...")
    models_day_cem, models_day_ring = train_stratified_day_models(
        train_val, loro_folds, n_trials=10, seed=42,
    )
    cal_cem = train_val[train_val["estimation_method"] == "cem_weighted"].sample(
        frac=0.15, random_state=42,
    )
    test_cem = test[test["estimation_method"] == "cem_weighted"]
    if len(test_cem) >= 3:
        diagnose_ci_coverage(
            models_day_cem[0], models_day_cem[1], models_day_cem[2],
            cal_cem, test_cem, "label_delta_t_day", "DAY CEM"
        )


if __name__ == "__main__":
    main()
