from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.fft import rfft


class DataWindowing:
    def __init__(self, window_size: int = 1000,
                 window_step: int = 1,
                 window_labeling_strategy: LabelStrategy | None = None,
                 ):
        self.window_size = window_size
        self.labeling_strategy: LabelStrategy = window_labeling_strategy or LabelMajorityStrategy()
        self.window_step = window_step
        self._validate()

    def windowGenerator(self, X: np.ndarray, y: np.ndarray, batch_size=None):

        if not isinstance(X, np.ndarray):
            raise ValueError(f'tablica X musi byc typu np.ndarray otrzymano {type(X)}')

        if not isinstance(y, np.ndarray):
            raise ValueError(f'tablica y musi byc typu np.ndarray otrzymano {type(y)}')

        if X.ndim != 2:
            raise ValueError(f'tablica X musi byc 2 wymiarowa, otrzymano {X.ndim}')

        if y.ndim != 1:
            raise ValueError(f'tablica y musi byc 1 wymiarowa, otrzymano {y.ndim}')

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"DataFrame'y są różnych rozmiarów: "
                f"data_df={X.shape[0]}, label_df={y.shape[0]}"
            )
        windows_X = []
        windows_y = []

        for start_idx in range(0, len(X) - self.window_size + 1, self.window_step):
            end_idx = start_idx + self.window_size

            win = X[start_idx:end_idx]
            label_window = y[start_idx:end_idx]

            window_label = self.labeling_strategy.get_label(label_window)

            if batch_size:
                windows_X.append(win)
                windows_y.append(window_label)

                if len(windows_X) == batch_size:
                    yield np.array(windows_X), np.array(windows_y)
                    windows_X = []
                    windows_y = []

            else:
                yield win, window_label

        # flush resztki batcha
        if batch_size and len(windows_X) > 0:
            yield np.array(windows_X), np.array(windows_y)

    def window(self, datas: np.ndarray, labels: np.ndarray = None):

        if not isinstance(datas, np.ndarray):
            raise ValueError(f'X musi byc typu np.ndarray, otrzymano {type(datas)}')

        if labels is not None and not isinstance(labels, np.ndarray):
            raise ValueError(f'y musi byc typu np.ndarray, otrzymano {type(labels)}')

        if labels is not None and datas.shape[0] != labels.shape[0]:
            raise ValueError(
                f"dane są różnych rozmiarów: "
                f"data={datas.shape[0]}, label={labels.shape[0]}"
            )

        X_win = []
        y_win = []
        for start_idx in range(0, len(datas) - self.window_size + 1, self.window_step):
            end_idx = start_idx + self.window_size

            win = datas[start_idx:end_idx]
            if labels is not None:
                label_window = labels[start_idx:end_idx]
                window_label = self.labeling_strategy.get_label(label_window)
                y_win.append(window_label)

            X_win.append(win)
        if labels is not None:
            return np.array(X_win), np.array(y_win)

        return np.array(X_win)

    def _validate(self):
        if self.window_size <= 0:
            raise ValueError("window_size musi być większe od 0")

        if self.window_step <= 0:
            raise ValueError("window_step musi być większe od 0")

    def window_with_extractions(self, datas: np.ndarray, labels: np.ndarray):

        windowed_data, windowed_label = self.window(datas, labels)

        extracted = extractWindowFeatures(windowed_data)
        return extracted, windowed_label

    def reverse_window_labels(self, windows, y_len):
        """
        Rozwija etykiety okien z powrotem na pojedyncze probki.

        Miejsce zapisu zalezy od strategii etykietowania: CenterLabelingStrategy
        opisuje srodek okna, wiec etykieta idzie na srodek, a nie na jego poczatek
        (inaczej caly wynik jest przesuniety o polowe okna).
        """
        restored = np.zeros(y_len)

        if len(windows) == 0:
            return restored

        offset = self.labeling_strategy.label_offset(self.window_size)
        span = self.window_step if offset else self.window_size

        for index, window in enumerate(windows):
            start = index * self.window_step + offset
            restored[start:start + span] = window

        # glowa i ogon sygnalu, ktorych nie pokrywa zadne okno
        last_covered = min(y_len, (len(windows) - 1) * self.window_step + offset + span)
        restored[:offset] = restored[offset] if offset < y_len else 0
        restored[last_covered:] = restored[last_covered - 1]

        return restored

    def get_params(self):
        return {
            'window_size': self.window_size, 'window_step': self.window_step
        }


def extractWindowFeatures(windows):
    if not isinstance(windows, np.ndarray):
        raise ValueError(f'windows musi byc typu np.ndarray a jest {type(windows)}')
    if windows.ndim != 3:
        raise ValueError('windows musi byc 3 wymiarowa [n_windows,rows,cols]')

    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    min_ = windows.min(axis=1)
    max_ = windows.max(axis=1)
    median = np.median(windows, axis=1)
    sq = np.square(windows).sum(axis=1)
    # _skew = skew(windows, axis=1)
    dominant_freq = np.argmax(np.abs(rfft(windows, axis=1)), axis=1)

    full_list = [mean, std, min_, max_, median, sq, dominant_freq]

    return np.concatenate(full_list, axis=1)


class LabelStrategy(ABC):

    @abstractmethod
    def get_label(self, windowed_labels):
        pass

    def label_offset(self, window_size: int) -> int:
        """Przesuniecie probki, ktora opisuje etykieta okna, wzgledem jego poczatku."""
        return 0


class LabelMajorityStrategy(LabelStrategy):
    def __init__(self, mainlabel=1, threshold=0.7):
        self.main_label = mainlabel
        self.threshold = threshold

    def get_label(self, windowed_labels):
        return int(np.mean(windowed_labels == self.main_label) >= self.threshold)


# def label_majority_strategy(windowed_labels: np.ndarray, main_label=1, threshold=0.7):
#     return int(np.mean(windowed_labels == main_label) >= threshold)


class CenterLabelingStrategy(LabelStrategy):

    def label_offset(self, window_size: int) -> int:
        return window_size // 2

    def get_label(self, windowed_labels):
        half = len(windowed_labels) // 2
        labeled = windowed_labels[half]

        return labeled
