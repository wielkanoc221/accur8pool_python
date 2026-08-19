from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class FilesManager:
    def __init__(
            self,
            root_dir: Optional[str | Path] = None,
            data_paths: Optional[Iterable[str | Path]] = None,
            labels_paths: Optional[Iterable[str | Path]] = None,
    ):
        self.root_dir = Path(root_dir) if root_dir else None

        self.data_paths = (
            [Path(p) for p in data_paths] if data_paths is not None else None
        )
        self.labels_paths = (
            [Path(p) for p in labels_paths] if labels_paths is not None else None
        )

        self._validate()

    def _validate(self):
        if self.root_dir is None and self.data_paths is None:
            raise ValueError(
                "Musisz podać root_dir albo data_paths + labels_paths"
            )

        if self.root_dir is not None:
            if not self.root_dir.exists():
                raise FileNotFoundError(f"{self.root_dir} nie istnieje")
            if not self.root_dir.is_dir():
                raise NotADirectoryError(f"{self.root_dir} nie jest katalogiem")

    def _resolve_data_paths(self):
        if self.data_paths is not None:
            return self.data_paths

        return list((self.root_dir / "prepared").rglob("*.csv"))

    def _resolve_labels_paths(self):
        if self.labels_paths is not None:
            return self.labels_paths

        return list((self.root_dir / "labeled").rglob("*.csv"))

    def get_data_paths(self):
        return self._resolve_data_paths()

    def get_labels_paths(self):
        return self._resolve_labels_paths()

    def get_data_label_paths(self) -> list[tuple[Path,Path]]:
        data_paths = self._resolve_data_paths()
        labels_paths = self._resolve_labels_paths()

        label_map = {p.name: p for p in labels_paths}

        return [
            (d, label_map[d.name])
            for d in data_paths
            if d.name in label_map
        ]

    def get_unlabeled_data(self):
        data_paths = self._resolve_data_paths()
        labels_paths = self._resolve_labels_paths()

        label_names = {p.name for p in labels_paths}

        return [
            d for d in data_paths
            if d.name not in label_names
        ]

    @staticmethod
    def get_csv_paths(dir_path: str | Path):
        path = Path(dir_path)

        if not path.is_dir():
            raise ValueError(f"{dir_path} nie jest katalogiem")

        return list(path.rglob("*.csv"))


if __name__ == '__main__':
    fm = FilesManager(root_dir=r'/data1/data1')
    x = fm.get_data_label_paths()
    print(x)
    for d, l in x:
        data = pd.read_csv(d)
        labels = pd.read_csv(l)
        if len(data) != len(labels):
            print(d, len(data), len(labels))
