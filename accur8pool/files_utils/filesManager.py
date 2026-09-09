import os
from functools import cached_property
from pathlib import Path
from typing import Iterable, Optional

DEFAULT_PATTERNS: tuple[str, ...] = ("*.csv", "*.CSV")


class FilesManager:
    """
    Zarzadza sciezkami do plikow z danymi oraz do plikow z etykietami.

    Sciezki mozna podac na dwa sposoby:
      * root_dir - katalog zawierajacy podkatalogi `data_subdir` i `labels_subdir`,
      * data_paths + labels_paths - jawne listy plikow (oba naraz).

    Plik z danymi jest laczony z etykietami po sciezce wzglednej (bez rozszerzenia)
    liczonej wzgledem katalogu bazowego grupy - `root_dir/prepared` i `root_dir/labeled`,
    a dla jawnych list wspolnego przedrostka kazdej z nich. Dzieki temu
    `prepared/s1/run.csv` nie jest mylone z `prepared/s2/run.csv` (samo porownanie
    nazwy plikow parowalo je bledno). Wymaga to, by obie struktury katalogow byly
    lustrzane - pliki bez odpowiednika trafiaja do `get_unlabeled_data()`.

    Wyniki skanowania katalogow sa cache'owane i posortowane, dzieki czemu
    kolejnosc plikow jest deterministyczna (ma to znaczenie m.in. dla `file_id`
    nadawanego w PandasDataFrameLoader.concat_datas_with_labels).
    """

    def __init__(
            self,
            root_dir: Optional[str | Path] = None,
            data_paths: Optional[Iterable[str | Path]] = None,
            labels_paths: Optional[Iterable[str | Path]] = None,
            data_subdir: str | Path = "prepared",
            labels_subdir: str | Path = "labeled",
            patterns: str | Iterable[str] = DEFAULT_PATTERNS,
    ):
        self.root_dir = Path(root_dir) if root_dir else None

        self.data_paths = (
            [Path(p) for p in data_paths] if data_paths is not None else None
        )
        self.labels_paths = (
            [Path(p) for p in labels_paths] if labels_paths is not None else None
        )

        self.data_subdir = data_subdir
        self.labels_subdir = labels_subdir
        self.patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)

        self._validate()

    def _validate(self) -> None:
        if self.root_dir is None and (self.data_paths is None or self.labels_paths is None):
            raise ValueError(
                "Musisz podac root_dir albo data_paths + labels_paths (oba naraz)"
            )

        if self.root_dir is not None:
            if not self.root_dir.exists():
                raise FileNotFoundError(f"{self.root_dir} nie istnieje")
            if not self.root_dir.is_dir():
                raise NotADirectoryError(f"{self.root_dir} nie jest katalogiem")

        for name, paths in (("data_paths", self.data_paths), ("labels_paths", self.labels_paths)):
            if paths is None:
                continue
            for path in paths:
                if not path.exists():
                    raise FileNotFoundError(f"{path} ({name}) nie istnieje")
                if not path.is_file():
                    raise ValueError(f"{path} ({name}) nie jest plikiem")

    def _scan(self, subdir: str | Path) -> list[Path]:
        directory = self.root_dir / subdir

        if not directory.is_dir():
            raise FileNotFoundError(f"{directory} nie istnieje lub nie jest katalogiem")

        # dict zamiast set - zachowuje kolejnosc i usuwa duplikaty, ktore powstaja
        # gdy kilka wzorcow trafia w ten sam plik na systemie plikow bez
        # rozroznienia wielkosci liter (Windows, macOS)
        found: dict[Path, None] = {}
        for pattern in self.patterns:
            for path in directory.rglob(pattern):
                if path.is_file():
                    found[path] = None

        return sorted(found)

    @cached_property
    def _resolved_data_paths(self) -> list[Path]:
        if self.data_paths is not None:
            return sorted(self.data_paths)

        return self._scan(self.data_subdir)

    @cached_property
    def _resolved_labels_paths(self) -> list[Path]:
        if self.labels_paths is not None:
            return sorted(self.labels_paths)

        return self._scan(self.labels_subdir)

    def refresh(self) -> None:
        """Czysci cache sciezek - do uzycia gdy pliki dochodza w trakcie zycia obiektu."""
        self.__dict__.pop("_resolved_data_paths", None)
        self.__dict__.pop("_resolved_labels_paths", None)

    @staticmethod
    def _common_base(paths: list[Path]) -> Optional[Path]:
        if not paths:
            return None
        if len(paths) == 1:
            return paths[0].parent
        try:
            base = Path(os.path.commonpath(paths))
            # commonpath dla powtorzonych sciezek zwraca sam plik, nie katalog
            return base.parent if base in set(paths) else base
        except ValueError:
            # mieszanka sciezek wzglednych i bezwzglednych - brak wspolnej bazy
            return None

    @staticmethod
    def _key(path: Path, base: Optional[Path]) -> str:
        rel = Path(path.name)
        if base is not None:
            try:
                relative = path.relative_to(base)
                if relative.name:
                    rel = relative
            except ValueError:
                pass

        return rel.with_suffix("").as_posix()

    def _index(self, paths: list[Path], base: Optional[Path], kind: str) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in paths:
            key = self._key(path, base)
            if key in index:
                raise ValueError(
                    f"Zduplikowany klucz {kind} '{key}': {index[key]} oraz {path}"
                )
            index[key] = path

        return index

    def _data_base(self) -> Optional[Path]:
        if self.data_paths is not None:
            return self._common_base(self._resolved_data_paths)

        return self.root_dir / self.data_subdir

    def _labels_base(self) -> Optional[Path]:
        if self.labels_paths is not None:
            return self._common_base(self._resolved_labels_paths)

        return self.root_dir / self.labels_subdir

    def _pair(self) -> tuple[list[tuple[Path, Path]], list[Path]]:
        """Zwraca (pary data-label, dane bez etykiet) - jeden przebieg indeksowania."""
        data_paths = self._resolved_data_paths
        labels_index = self._index(self._resolved_labels_paths, self._labels_base(), "etykiet")
        self._index(data_paths, self._data_base(), "danych")

        pairs: list[tuple[Path, Path]] = []
        unlabeled: list[Path] = []
        data_base = self._data_base()

        for data_path in data_paths:
            label_path = labels_index.get(self._key(data_path, data_base))
            if label_path is None:
                unlabeled.append(data_path)
            else:
                pairs.append((data_path, label_path))

        return pairs, unlabeled

    def get_data_paths(self) -> list[Path]:
        return list(self._resolved_data_paths)

    def get_labels_paths(self) -> list[Path]:
        return list(self._resolved_labels_paths)

    def get_data_label_paths(self) -> list[tuple[Path, Path]]:
        return self._pair()[0]

    def get_unlabeled_data(self) -> list[Path]:
        return self._pair()[1]

    @staticmethod
    def get_csv_paths(dir_path: str | Path, patterns: str | Iterable[str] = DEFAULT_PATTERNS) -> list[Path]:
        path = Path(dir_path)

        if not path.is_dir():
            raise ValueError(f"{dir_path} nie jest katalogiem")

        patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)

        found: dict[Path, None] = {}
        for pattern in patterns:
            for file_path in path.rglob(pattern):
                if file_path.is_file():
                    found[file_path] = None

        return sorted(found)


if __name__ == '__main__':
    import sys

    import pandas as pd

    root = sys.argv[1] if len(sys.argv) > 1 else '/data1/data1'
    fm = FilesManager(root_dir=root)
    pairs = fm.get_data_label_paths()
    print(f"{len(pairs)} par, {len(fm.get_unlabeled_data())} plikow bez etykiet")

    for data_path, labels_path in pairs:
        data = pd.read_csv(data_path)
        labels = pd.read_csv(labels_path)
        if len(data) != len(labels):
            print(data_path, len(data), len(labels))
