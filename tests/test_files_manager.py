import pytest

from accur8pool.files_utils.filesManager import FilesManager


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("a,b\n1,2\n")
    return path


@pytest.fixture
def root(tmp_path):
    for name in ("s1/run.csv", "s2/run.csv", "flat.csv"):
        _touch(tmp_path / "prepared" / name)
    for name in ("s1/run.csv", "s2/run.csv"):
        _touch(tmp_path / "labeled" / name)

    return tmp_path


def test_requires_root_or_both_path_lists(tmp_path):
    with pytest.raises(ValueError):
        FilesManager()

    with pytest.raises(ValueError):
        FilesManager(data_paths=[_touch(tmp_path / "a.csv")])


def test_missing_root_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        FilesManager(root_dir=tmp_path / "nope")


def test_root_dir_must_be_a_directory(tmp_path):
    with pytest.raises(NotADirectoryError):
        FilesManager(root_dir=_touch(tmp_path / "a.csv"))


def test_explicit_paths_must_exist(tmp_path):
    with pytest.raises(FileNotFoundError):
        FilesManager(data_paths=[tmp_path / "nope.csv"], labels_paths=[])


def test_missing_subdir_raises(tmp_path):
    (tmp_path / "prepared").mkdir()
    fm = FilesManager(root_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        fm.get_labels_paths()


def test_paths_are_sorted_and_deterministic(root):
    fm = FilesManager(root_dir=root)
    paths = fm.get_data_paths()

    assert paths == sorted(paths)
    assert [p.name for p in paths] == ["flat.csv", "run.csv", "run.csv"]


def test_pairing_uses_relative_path_not_filename(root):
    pairs = FilesManager(root_dir=root).get_data_label_paths()

    assert len(pairs) == 2
    for data_path, labels_path in pairs:
        assert data_path.relative_to(root / "prepared") == labels_path.relative_to(root / "labeled")


def test_unlabeled_data(root):
    assert [p.name for p in FilesManager(root_dir=root).get_unlabeled_data()] == ["flat.csv"]


def test_ambiguous_labels_are_not_paired_silently(tmp_path):
    """Te same nazwy plikow w roznych katalogach nie moga byc sparowane na slepo."""
    data = [_touch(tmp_path / "d" / "a.csv")]
    labels = [_touch(tmp_path / "l1" / "a.csv"), _touch(tmp_path / "l2" / "a.csv")]

    fm = FilesManager(data_paths=data, labels_paths=labels)

    assert fm.get_data_label_paths() == []
    assert fm.get_unlabeled_data() == data


def test_duplicate_keys_raise(root):
    """Dwa pliki danych o identycznej sciezce wzglednej wzgledem bazy -> blad."""
    fm = FilesManager(root_dir=root)
    fm.data_paths = [root / "prepared" / "s1" / "run.csv"] * 2
    fm.refresh()

    with pytest.raises(ValueError, match="Zduplikowany klucz"):
        fm.get_data_label_paths()


def test_flat_explicit_lists_pair_by_stem(tmp_path):
    data = [_touch(tmp_path / "d" / f"{n}.csv") for n in ("a", "b")]
    labels = [_touch(tmp_path / "l" / "a.csv")]

    fm = FilesManager(data_paths=data, labels_paths=labels)

    assert fm.get_data_label_paths() == [(data[0], labels[0])]
    assert fm.get_unlabeled_data() == [data[1]]


def test_scan_is_cached_until_refresh(root):
    fm = FilesManager(root_dir=root)
    assert len(fm.get_data_paths()) == 3

    _touch(root / "prepared" / "s3" / "run.csv")
    assert len(fm.get_data_paths()) == 3

    fm.refresh()
    assert len(fm.get_data_paths()) == 4


def test_returned_lists_are_copies(root):
    fm = FilesManager(root_dir=root)
    fm.get_data_paths().clear()

    assert len(fm.get_data_paths()) == 3


def test_custom_subdirs_and_patterns(tmp_path):
    _touch(tmp_path / "in" / "a.txt")
    _touch(tmp_path / "out" / "a.txt")

    fm = FilesManager(root_dir=tmp_path, data_subdir="in", labels_subdir="out", patterns="*.txt")

    assert len(fm.get_data_label_paths()) == 1


def test_uppercase_extension_is_found(tmp_path):
    _touch(tmp_path / "prepared" / "a.CSV")
    (tmp_path / "labeled").mkdir()

    assert [p.name for p in FilesManager(root_dir=tmp_path).get_data_paths()] == ["a.CSV"]


def test_get_csv_paths(root):
    assert len(FilesManager.get_csv_paths(root / "prepared")) == 3

    with pytest.raises(ValueError):
        FilesManager.get_csv_paths(root / "nope")
