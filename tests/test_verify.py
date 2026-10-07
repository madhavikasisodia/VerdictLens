from verdictlens.verify import verify_summary


def test_verification_detects_unsupported_and_missing_entities() -> None:
    source = "The Supreme Court allowed the appeal under Article 21."
    report = verify_summary(source, "The High Court dismissed the appeal.")
    assert "high court" in report.unsupported_entities
    assert report.outcome_inconsistencies == ["dismissed"]
    assert not report.is_faithful


def test_matching_summary_is_faithful() -> None:
    source = "The Supreme Court allowed the appeal under Article 21."
    assert verify_summary(source, source).is_faithful


def test_structured_metrics_and_foreign_leakage() -> None:
    source = "The Supreme Court considered Section 302 IPC and Article 21. The appeal was allowed."
    summary = "The Supreme Court considered Section 302 IPC. The Supreme Court of the United States allowed it under U.S.C."
    report = verify_summary(source, summary, p3_outcome="allowed", check_support=False)
    assert report.entity_precision == 1.0
    assert report.statute_precision == 1.0
    assert report.statute_recall < 1.0
    assert "u.s.c" in report.foreign_leakage
    assert report.outcome_match is True
    assert report.support_model_status == "disabled"


def test_p3_outcome_mismatch_is_reported() -> None:
    source = "The appeal was allowed."
    report = verify_summary(source, "The appeal was dismissed.", p3_outcome="allowed", check_support=False)
    assert report.outcome_match is False
    assert report.outcome_inconsistencies == ["dismissed"]
