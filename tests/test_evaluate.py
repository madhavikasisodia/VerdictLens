from verdictlens.evaluate import bootstrap_mean_ci, evaluate_summary, optional_rouge, paired_permutation_test, token_f1


def test_token_f1_and_empty_cases() -> None:
    assert token_f1("a b", "a b") == 1.0
    assert token_f1("", "a") == 0.0


def test_evaluation_reports_faithfulness() -> None:
    metrics = evaluate_summary("The court allowed the appeal.", "The court allowed the appeal.", "The court allowed the appeal.")
    assert metrics["faithful"] is True
    assert optional_rouge("a", "a") is None or "rouge1" in optional_rouge("a", "a")


def test_bootstrap_and_paired_tests_are_deterministic() -> None:
    first = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=42, samples=100)
    second = bootstrap_mean_ci([1.0, 2.0, 3.0], seed=42, samples=100)
    assert first == second
    paired = paired_permutation_test([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], seed=42, samples=100)
    assert paired["n"] == 3.0
    assert paired["mean_difference"] == 1.0
