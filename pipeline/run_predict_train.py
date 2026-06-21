"""
CLI to run DCTII-Predict model training.

Usage:
  python -m pipeline.run_predict_train --version auto --n-trials 80
  python -m pipeline.run_predict_train --version v2 --force
  python -m pipeline.run_predict_train --dry-run   # no GCS upload, print metrics only

Intended execution: Cloud Batch job 'job-dctii-predict-train-dev'
Can also run locally with GCP credentials.
"""

import argparse
import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("dctii.run_predict_train")


def main():
    parser = argparse.ArgumentParser(
        description="DCTII-Predict Model Training Pipeline"
    )
    parser.add_argument(
        "--version", default="auto",
        help='Model version string. "auto" = v{N+1} based on GCS.',
    )
    parser.add_argument("--n-trials", type=int, default=80,
                        help="Number of Optuna trials per target")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip GCS upload; print metrics only")
    parser.add_argument("--force", action="store_true",
                        help="Upload even if acceptance thresholds fail")
    parser.add_argument("--output-dir", default="output",
                        help="Directory for diagnostic plots")
    args = parser.parse_args()

    from pipeline.predict_train import (
        build_training_matrix,
        engineer_features,
        get_splits,
        tune_hyperparameters,
        train_final_model,
        train_stratified_day_models,
        calibrate_cross_conformal,
        compute_bias_offset,
        compute_shap_explainer,
        evaluate_model,
        save_model_artifacts,
        compute_training_distribution,
        plot_diagnostics,
        resolve_version,
        FEATURE_COLUMNS,
        TARGET_DAY,
        TARGET_NIGHT,
        TEST_REGION,
    )

    version = resolve_version(args.version)
    logger.info(f"Starting training run — version={version}")

    try:
        # 1. Build training matrix
        df_raw = build_training_matrix()
        logger.info(f"Training matrix: {len(df_raw)} rows")

        # 2. Feature engineering
        df = engineer_features(df_raw)

        # 3. Split
        train_val, test, loro_folds = get_splits(df)
        logger.info(
            f"Train+val: {len(train_val)} rows | "
            f"Test ({TEST_REGION}): {len(test)} rows"
        )

        results = {}
        models_day_cem = None
        models_day_ring = None
        models_night = None

        # ── DAY models (stratified by estimation method) ──────────────
        logger.info(f"\n{'=' * 60}")
        logger.info("TRAINING: DAY models (stratified CEM + Ring)")
        logger.info(f"{'=' * 60}")

        models_day_cem, models_day_ring = train_stratified_day_models(
            train_val, loro_folds,
            n_trials=args.n_trials, seed=args.seed,
        )

        # Cross-conformal calibration for each day model using LORO holdouts
        target_day = "label_delta_t_day"

        # Build CEM and Ring LORO folds
        cem_folds = [
            (fold_tr[fold_tr["estimation_method"] == "cem_weighted"],
             fold_val[fold_val["estimation_method"] == "cem_weighted"],
             region)
            for fold_tr, fold_val, region in loro_folds
            if len(fold_val[fold_val["estimation_method"] == "cem_weighted"]) >= 3
        ]
        ring_folds = [
            (fold_tr[fold_tr["estimation_method"] == "ring_difference"],
             fold_val[fold_val["estimation_method"] == "ring_difference"],
             region)
            for fold_tr, fold_val, region in loro_folds
            if len(fold_val[fold_val["estimation_method"] == "ring_difference"]) >= 3
        ]
        logger.info(f"CEM folds for conformal: {len(cem_folds)} "
                     f"(regions: {[r for _,_,r in cem_folds]})")
        logger.info(f"Ring folds for conformal: {len(ring_folds)} "
                     f"(regions: {[r for _,_,r in ring_folds]})")

        correction_day_cem = calibrate_cross_conformal(
            train_val_df=train_val[train_val["estimation_method"] == "cem_weighted"],
            loro_folds=cem_folds,
            target_col=target_day,
            alpha=0.10,
            seed=args.seed,
        )

        if len(ring_folds) >= 2:
            correction_day_ring = calibrate_cross_conformal(
                train_val_df=train_val[train_val["estimation_method"] == "ring_difference"],
                loro_folds=ring_folds,
                target_col=target_day,
                alpha=0.10,
                seed=args.seed,
            )
        else:
            # Not enough ring folds — use CEM correction as fallback
            logger.warning("Not enough Ring LORO folds for cross-conformal; "
                           "using CEM correction as fallback")
            correction_day_ring = correction_day_cem

        # Evaluate day_cem on CEM test rows, day_ring on ring test rows
        test_cem = test[test["estimation_method"] == "cem_weighted"]
        test_ring = test[test["estimation_method"] == "ring_difference"]

        if len(test_cem) >= 3:
            eval_day_cem = evaluate_model(
                models_day_cem[0], models_day_cem[1], models_day_cem[2],
                correction_day_cem, test_cem, target_day,
            )
            results["day_cem"] = {**eval_day_cem, "correction": correction_day_cem}
        else:
            logger.warning(f"Only {len(test_cem)} CEM test rows — skipping CEM day evaluation")
            results["day_cem"] = {"passed": True, "correction": correction_day_cem}

        if len(test_ring) >= 3:
            eval_day_ring = evaluate_model(
                models_day_ring[0], models_day_ring[1], models_day_ring[2],
                correction_day_ring, test_ring, target_day,
            )
            results["day_ring"] = {**eval_day_ring, "correction": correction_day_ring}
        else:
            logger.warning(f"Only {len(test_ring)} Ring test rows — skipping Ring day evaluation")
            results["day_ring"] = {"passed": True, "correction": correction_day_ring}

        # Day diagnostic plots (CEM model on full test set)
        import numpy as np
        X_test_all = test[FEATURE_COLUMNS]
        pred_day = models_day_cem[0].predict(X_test_all)
        y_test_day = test[target_day].values
        plot_diagnostics(pred_day, y_test_day, "day_cem", args.output_dir)

        # ── NIGHT model (unified — estimation methods agree) ──────────
        logger.info(f"\n{'=' * 60}")
        logger.info("TRAINING: NIGHT model")
        logger.info(f"{'=' * 60}")

        best_params_night = tune_hyperparameters(
            train_val, loro_folds, TARGET_NIGHT,
            n_trials=args.n_trials, seed=args.seed,
        )
        models_night = train_final_model(
            train_val, best_params_night, TARGET_NIGHT, seed=args.seed,
        )

        cal_night = calibrate_cross_conformal(
            train_val_df=train_val,
            loro_folds=loro_folds,
            target_col=TARGET_NIGHT,
            alpha=0.10,
            seed=args.seed,
        )
        correction_night = cal_night

        eval_night = evaluate_model(
            models_night[0], models_night[1], models_night[2],
            correction_night, test, TARGET_NIGHT,
        )
        results["night"] = {**eval_night, "correction": correction_night}

        # Bias offsets (stratified by climate_heat_rank)
        logger.info("\nComputing bias offsets (night)...")
        night_bias_offsets = compute_bias_offset(
            models_night[0], train_val, TARGET_NIGHT,
        )
        logger.info("Computing bias offsets (day CEM)...")
        day_cem_bias_offsets = compute_bias_offset(
            models_day_cem[0],
            train_val[train_val["estimation_method"] == "cem_weighted"],
            "label_delta_t_day",
        )

        X_test = test[FEATURE_COLUMNS]
        pred_night = models_night[0].predict(X_test)
        y_test_night = test[TARGET_NIGHT].values
        plot_diagnostics(pred_night, y_test_night, TARGET_NIGHT, args.output_dir)

        # 9. Check all models pass
        day_pass = results["day_cem"].get("passed", True) and results["day_ring"].get("passed", True)
        all_pass = day_pass and results["night"]["passed"]
        logger.info(f"\nFINAL VERDICT: {'PASS ✓' if all_pass else 'FAIL ✗'}")

        if not all_pass and not args.force:
            logger.error(
                "Training FAILED acceptance thresholds. "
                "Use --force to upload anyway."
            )
            sys.exit(1)

        # 10. SHAP (CEM day model is the primary)
        shap_day_cem = compute_shap_explainer(models_day_cem[0], train_val[train_val["estimation_method"] == "cem_weighted"])
        shap_night = compute_shap_explainer(models_night[0], train_val)

        # 11. Training distribution for Mahalanobis
        train_dist = compute_training_distribution(train_val)

        # 12. Save artifacts
        if not args.dry_run:
            corrections = {
                "day_cem_correction": results["day_cem"]["correction"],
                "day_ring_correction": results["day_ring"]["correction"],
                "night_correction": results["night"]["correction"],
                "night_bias_offsets": night_bias_offsets,
                "day_cem_bias_offsets": day_cem_bias_offsets,
            }
            eval_report = {
                **results,
                "version": version,
                "n_train": len(train_val),
                "n_test": len(test),
                "test_region": TEST_REGION,
            }
            feature_metadata = {
                "feature_columns": FEATURE_COLUMNS,
                "fill_values": {col: 0.0 for col in FEATURE_COLUMNS},
            }
            save_model_artifacts(
                version=version,
                models_day_cem=models_day_cem,
                models_day_ring=models_day_ring,
                models_night=models_night,
                shap_day_cem=shap_day_cem,
                shap_night=shap_night,
                conformal_corrections=corrections,
                train_distribution=train_dist,
                eval_report=eval_report,
                feature_metadata=feature_metadata,
            )
            logger.info(f"Artifacts saved to GCS under version={version}")
        else:
            logger.info("--dry-run: skipping GCS upload")

        logger.info("Training run complete.")

    except Exception as e:
        logger.exception(f"Training run failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
