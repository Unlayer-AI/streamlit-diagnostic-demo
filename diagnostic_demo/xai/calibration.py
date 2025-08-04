from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import brier_score_loss


def get_calibration_metrics(model, df, target_variable, calibset_size=0.5):
    # Make up a calib set
    X_temp, y_temp = df.drop(columns=[target_variable]), df[target_variable]
    X_calib, X_test, y_calib, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - calibset_size, random_state=42
    )

    # Get model predictions
    uncalibrated_probs = model.predict_proba(X_calib)[:, 1]

    # Apply Isotonic Regression for Calibration
    iso_reg = IsotonicRegression(
        out_of_bounds="clip"
    )  # Ensure values stay within [0,1]
    iso_reg.fit(uncalibrated_probs, y_calib)

    uncalibrated_test_probs = model.predict_proba(X_test)[:, 1]
    calibrated_test_probs = iso_reg.transform(uncalibrated_test_probs)

    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(
        y_test, uncalibrated_test_probs, n_bins=20
    )
    prob_true_iso, prob_pred_iso = calibration_curve(
        y_test, calibrated_test_probs, n_bins=20
    )

    # Compute new calibration curve
    uncalibrated_brier = brier_score_loss(y_test, uncalibrated_test_probs)
    calibrated_brier = brier_score_loss(y_test, calibrated_test_probs)

    return {
        "calib_prob_true": prob_true,
        "calib_prob_pred": prob_pred,
        "calib_prob_true_iso": prob_true_iso,
        "calib_prob_pred_iso": prob_pred_iso,
        "calib_brier": uncalibrated_brier,
        "calib_brier_iso": calibrated_brier,
    }
