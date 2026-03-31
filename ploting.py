from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ShotEvent:
    start_idx: int
    end_idx: int
    peak_idx: int
    peak_linacc: float
    peak_gyr: float
    duration_ms: float
    energy_linacc: float


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    idx = (len(sorted_values) - 1) * p
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[lo]
    return sorted_values[lo] * (hi - idx) + sorted_values[hi] * (idx - lo)


def vector_magnitude(x: float, y: float, z: float) -> float:
    return math.sqrt(x * x + y * y + z * z)


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in reader]


def detect_shots(
    linacc_mag: list[float],
    timestamps_ms: list[float],
    threshold: float,
    merge_gap_ms: float = 80.0,
) -> list[tuple[int, int]]:
    raw_segments: list[tuple[int, int]] = []
    in_segment = False
    start = 0

    for i, value in enumerate(linacc_mag):
        if value >= threshold and not in_segment:
            start = i
            in_segment = True
        elif in_segment and value < threshold:
            raw_segments.append((start, i - 1))
            in_segment = False

    if in_segment:
        raw_segments.append((start, len(linacc_mag) - 1))

    if not raw_segments:
        return []

    merged: list[list[int]] = [[raw_segments[0][0], raw_segments[0][1]]]
    current_gap = 0.0
    for start, end in raw_segments[1:]:
        prev_end = merged[-1][1]
        current_gap = sum(timestamps_ms[prev_end + 1 : start + 1])
        if current_gap <= merge_gap_ms:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    return [(s, e) for s, e in merged]


def summarize_shots(
    shot_ranges: Iterable[tuple[int, int]],
    linacc_mag: list[float],
    gyr_mag: list[float],
    timestamps_ms: list[float],
) -> list[ShotEvent]:
    events: list[ShotEvent] = []

    for start, end in shot_ranges:
        peak_idx = max(range(start, end + 1), key=lambda i: linacc_mag[i])
        duration_ms = sum(timestamps_ms[start : end + 1])
        energy_linacc = 0.0
        for i in range(start, end + 1):
            dt = timestamps_ms[i] / 1000.0
            energy_linacc += (linacc_mag[i] ** 2) * dt

        events.append(
            ShotEvent(
                start_idx=start,
                end_idx=end,
                peak_idx=peak_idx,
                peak_linacc=linacc_mag[peak_idx],
                peak_gyr=max(gyr_mag[start : end + 1]),
                duration_ms=duration_ms,
                energy_linacc=energy_linacc,
            )
        )

    return events


def create_report(path: Path, events_to_show: int = 12) -> str:
    rows = load_rows(path)
    if not rows:
        return "Brak danych w pliku."

    timestamps = [row["timestamp"] for row in rows]
    linacc_mag = [vector_magnitude(row["linaccx"], row["linaccy"], row["linaccz"]) for row in rows]
    gyr_mag = [vector_magnitude(row["gyrx"], row["gyry"], row["gyrz"]) for row in rows]

    # W danych timestamp wygląda na interwał próbkowania w ms.
    # Pierwsza wartość jest często outlierem inicjalizacji, więc raportujemy statystyki bez niej.
    ts_body = timestamps[1:] if len(timestamps) > 1 else timestamps

    lin_med = statistics.median(linacc_mag)
    lin_mad = statistics.median([abs(x - lin_med) for x in linacc_mag])
    threshold = lin_med + 3.0 * lin_mad

    shot_ranges = detect_shots(linacc_mag, timestamps, threshold=threshold, merge_gap_ms=80.0)
    events = summarize_shots(shot_ranges, linacc_mag, gyr_mag, timestamps)

    strongest = sorted(events, key=lambda e: e.peak_linacc, reverse=True)[:events_to_show]
    average_event_duration = statistics.mean([e.duration_ms for e in events]) if events else 0.0

    total_time_s = sum(ts_body) / 1000.0

    lines: list[str] = []
    lines.append(f"# Analiza: {path.name}")
    lines.append("")
    lines.append("## 1) Podsumowanie sygnału")
    lines.append(f"- Liczba próbek: **{len(rows)}**")
    lines.append(f"- Szacowany czas nagrania (bez pierwszego outliera): **{total_time_s:.2f} s**")
    lines.append(
        f"- Interwał próbkowania [ms]: mediana **{statistics.median(ts_body):.3f}**, p95 **{percentile(ts_body, 0.95):.3f}**, max **{max(ts_body):.3f}**"
    )
    lines.append(
        f"- |linacc| [m/s²]: mediana **{lin_med:.3f}**, p95 **{percentile(linacc_mag, 0.95):.3f}**, p99 **{percentile(linacc_mag, 0.99):.3f}**, max **{max(linacc_mag):.3f}**"
    )
    lines.append(
        f"- |gyr| [rad/s]: mediana **{statistics.median(gyr_mag):.3f}**, p95 **{percentile(gyr_mag, 0.95):.3f}**, p99 **{percentile(gyr_mag, 0.99):.3f}**, max **{max(gyr_mag):.3f}**"
    )
    lines.append("")

    lines.append("## 2) Detekcja uderzeń")
    lines.append(
        f"- Próg detekcji (robust): **{threshold:.3f} m/s²** (`mediana + 3*MAD`)."
    )
    lines.append(f"- Wykryto zdarzeń (po scalaniu krótkich przerw): **{len(events)}**")
    lines.append(f"- Średni czas zdarzenia: **{average_event_duration:.1f} ms**")
    lines.append("")

    if strongest:
        lines.append("### Najmocniejsze zdarzenia")
        lines.append("")
        lines.append("| # | Zakres próbek | Czas [ms] | Peak |linacc| [m/s²] | Peak |gyr| [rad/s] | Energia linacc* [a²·s] |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for i, event in enumerate(strongest, start=1):
            lines.append(
                "| {i} | {s}-{e} | {d:.1f} | {pla:.3f} | {pg:.3f} | {en:.3f} |".format(
                    i=i,
                    s=event.start_idx,
                    e=event.end_idx,
                    d=event.duration_ms,
                    pla=event.peak_linacc,
                    pg=event.peak_gyr,
                    en=event.energy_linacc,
                )
            )
        lines.append("")

    lines.append("## 3) Co można z tym zrobić dalej")
    lines.append("1. **Ocena siły uderzenia**: użyć `peak |linacc|` albo `energia linacc` jako KPI mocy.")
    lines.append("2. **Ocena techniki**: porównywać `peak |gyr|` do mocy — duża rotacja przy małej mocy może wskazywać niestabilny chwyt/ruch.")
    lines.append("3. **Automatyczne tagowanie treningu**: wykryte eventy są gotowymi segmentami (strzałami) do późniejszej klasyfikacji typu uderzenia.")
    lines.append("4. **Model ML**: z każdego eventu wyliczyć cechy (czas narastania, symetria, energia, jerk) i trenować klasyfikator jakości uderzenia.")
    lines.append("5. **Feedback na żywo**: threshold + okno 100-300 ms pozwala robić sygnał „za słabo / idealnie / za mocno” podczas treningu.")
    lines.append("")
    lines.append("\* `energia linacc` to uproszczony wskaźnik: całka z |linacc|² po czasie; dobry do porównywania względnej dynamiki uderzeń.")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analiza danych uderzeń bilardowych (CSV IMU).")
    parser.add_argument("csv_path", type=Path, help="Ścieżka do pliku CSV")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Plik wyjściowy raportu Markdown (domyślnie obok CSV)",
    )
    args = parser.parse_args()
    path = Path(r'C:\Users\apietka\PycharmProjects\accur8pool\data\data20260329_220029.csv')
    report = create_report(path)
    output_path = args.output or args.csv_path.with_suffix(".analysis.md")
    output_path.write_text(report, encoding="utf-8")
    print(f"Zapisano raport: {output_path}")


if __name__ == "__main__":
    main()