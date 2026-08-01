from datetime import timedelta

import pytest

from paperbench.nano.utils import get_file_at_duration, load_paper_split


def test_load_paper_split_discovers_a_new_split_file(tmp_path) -> None:
    (tmp_path / "new-paper.txt").write_text("paper-one\npaper-two\n")

    assert load_paper_split("new-paper", splits_dir=tmp_path) == [
        "paper-one",
        "paper-two",
    ]


def test_load_paper_split_rejects_an_unknown_split(tmp_path) -> None:
    (tmp_path / "known.txt").write_text("paper-one\n")

    with pytest.raises(ValueError, match="Unknown paper split 'missing'.*known"):
        load_paper_split("missing", splits_dir=tmp_path)


@pytest.mark.parametrize("split_name", ["", "../all", "all.txt", "name/other"])
def test_load_paper_split_rejects_an_invalid_name(
    tmp_path, split_name: str
) -> None:
    with pytest.raises(ValueError, match="Invalid paper split name"):
        load_paper_split(split_name, splits_dir=tmp_path)


def test_load_paper_split_rejects_an_empty_split(tmp_path) -> None:
    (tmp_path / "empty.txt").write_text("\n")

    with pytest.raises(ValueError, match="contains no paper IDs"):
        load_paper_split("empty", splits_dir=tmp_path)


def test_get_file_at_duration_returns_expected_result() -> None:
    # Given
    files = [
        "/logs/run/2024-12-07T10-00-00-GMT/output.tar.gz",
        "/logs/run/2024-12-07T10-59-59-GMT/output.tar.gz",
        "/logs/run/2024-12-07T11-30-00-GMT/output.tar.gz",
    ]

    # When
    result = get_file_at_duration(files, 1)

    # Then
    assert result["path"] == "/logs/run/2024-12-07T10-59-59-GMT/output.tar.gz"
    assert result["duration"] == timedelta(hours=0, minutes=59, seconds=59)


def test_get_file_at_duration_raises_when_timestamp_missing() -> None:
    # Given
    files = ["/logs/run/latest/output.tar.gz"]

    # Then
    with pytest.raises(ValueError):
        get_file_at_duration(files, 1)


def test_get_file_at_duration_raises_with_short_path() -> None:
    # Given
    files = ["output.tar.gz"]

    # Then
    with pytest.raises(ValueError):
        get_file_at_duration(files, 1)
