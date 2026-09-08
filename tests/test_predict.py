"""Smoke tests for prediction pipeline."""
import csv
import tempfile

from src.predict import assess_risk, build_features_from_summary, recommend

THRESHOLDS = {"terminate_cpu": 5, "downsize_cpu": 20, "review_cpu": 50}
MARGINS = {
    "terminate": {"safe": 3, "moderate": 1},
    "downsize": {"safe": 8, "moderate": 3},
}


def test_recommend_terminate():
    assert recommend(pred_high=3.0, pred_mid=1.0, thresholds=THRESHOLDS) == "terminate"


def test_recommend_downsize():
    assert recommend(pred_high=15.0, pred_mid=10.0, thresholds=THRESHOLDS) == "downsize"


def test_recommend_review():
    assert recommend(pred_high=30.0, pred_mid=25.0, thresholds=THRESHOLDS) == "review"


def test_recommend_keep():
    assert recommend(pred_high=60.0, pred_mid=55.0, thresholds=THRESHOLDS) == "keep"


def test_risk_terminate_safe():
    assert assess_risk("terminate", 1.0, margins=MARGINS) == "safe"


def test_risk_terminate_risky():
    assert assess_risk("terminate", 4.5, margins=MARGINS) == "risky"


def test_risk_downsize_safe():
    assert assess_risk("downsize", 10.0, margins=MARGINS) == "safe"


def test_risk_keep_na():
    assert assess_risk("keep", 60.0, margins=MARGINS) == "n/a"


def test_build_features_with_std():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(["instance", "cpu_mean", "cpu_std", "cpu_min", "cpu_median", "cpu_max"])
        w.writerow(["vm-1", "10.0", "3.0", "2.0", "9.0", "20.0"])
        f.flush()
        instances, X = build_features_from_summary(f.name)

    assert instances == ["vm-1"]
    assert X.shape == (1, 5)
    assert X[0, 0] == 3.0  # std
    assert X[0, 1] == 2.0  # min
    assert X[0, 2] == 9.0  # p50


def test_build_features_without_std():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        w = csv.writer(f)
        w.writerow(["instance", "cpu_mean", "cpu_min", "cpu_median", "cpu_max"])
        w.writerow(["vm-1", "10.0", "2.0", "9.0", "20.0"])
        f.flush()
        instances, X = build_features_from_summary(f.name)

    assert instances == ["vm-1"]
    assert X[0, 0] == (20.0 - 2.0) / 4  # approximated std
