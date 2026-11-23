import numpy as np

from detector.eval.metrics import rates_at_threshold, threshold_at_fpr

def test_threshold_respects_fpr_budget():
    human = np.arange(100, dtype=float)     #100 human texts, scores 0..99
    ai = np.array([90.0, 95.0, 99.0, 50.0])
    y = np.array([0] * 100 + [1] * 4)
    s = np.concatenate([human, ai])

    thr = threshold_at_fpr(y, s, fpr=.05)
    rates = rates_at_threshold(y, s, thr)

    assert rates["fpr"] <= .05
    assert rates["tpr"] == 0.5      #only the AI scores 95 and 99 clear the cutoff

def test_threshold_budget_when_fpr_does_not_divide_evenly():
    rng = np.random.default_rng(0)
    y = np.array([0] * 150 + [1] * 20)
    s = np.concatenate([rng.random(150), rng.random(20) + .5])

    thr = threshold_at_fpr(y, s, fpr=.01)
    assert rates_at_threshold(y, s, thr)["fpr"] <= .01