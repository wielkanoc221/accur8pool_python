import argparse
import sys
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from accur8pool.data_processing.const import (
    ACC_MAGNITUDE,
    ACC_X,
    ACC_Y,
    ACC_Z,
    GYR_MAGNITUDE,
    GYR_X,
    GYR_Y,
    GYR_Z,
)
from accur8pool.data_processing.data_transformations import DataFrameTransformerBase, DataFrameTransformerV2
from accur8pool.files_utils.filesManager import FilesManager


class FileReadException(Exception):
    pass


class TransformException(Exception):
    pass


class SaveException(Exception):
    pass


class WrongColumnsException(Exception):
    pass


def transform_raw_df(df: DataFrame) -> pd.DataFrame:
    try:
        COLUMNS_TO_FILTER_10_CUT_OFF = ['accx', 'accy', 'accz', 'linaccx', 'linaccy', 'linaccz', ]
        COLUMNS_TO_FILTER_5_CUT_OFF = ['rotx', 'roty', 'rotz', 'gyrx', 'gyry', 'gyrz', 'magx', 'magy', 'magz']
        if 'csv_version' in df.columns:
            transformer = DataFrameTransformerV2

        else:
            transformer = DataFrameTransformerBase
        return (
            transformer(df)
            .dt_ms_to_sec()
            .lowpass(columns=COLUMNS_TO_FILTER_10_CUT_OFF, cutoff=10)
            .lowpass(columns=COLUMNS_TO_FILTER_5_CUT_OFF, cutoff=5)
            .add_magnitude([ACC_X, ACC_Y, ACC_Z], ACC_MAGNITUDE)
            .add_magnitude([GYR_X, GYR_Y, GYR_Z], GYR_MAGNITUDE)
            .add_time()
            .add_jerk([ACC_X, ACC_Y, ACC_Z], prefix='acc')
            .add_jerk([GYR_X, GYR_Y, GYR_Z], prefix='gyr')
            .add_roll()
            .add_pitch()
            .result()
        )
    except Exception as e:
        raise TransformException(e) from e


REQUIRED_COLUMNS = frozenset({
    'accx', 'accy', 'accz', 'gyrx', 'gyry', 'gyrz', 'magx', 'magy', 'magz',
    'linaccx', 'linaccy', 'linaccz', 'rotx', 'roty', 'rotz', 'timestamp',
})


def check_columns(df: pd.DataFrame) -> None:
    """Rzuca WrongColumnsException gdy brakuje ktorejs z wymaganych kolumn."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise WrongColumnsException(f'brakujace kolumny: {sorted(missing)}')


def save_data(df: DataFrame, output_dir, filename):
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / filename
        df.to_csv(save_path, index=False)
    except Exception as e:
        raise SaveException(e) from e


def read_csv(path):
    try:
        df = pd.read_csv(path, engine="pyarrow")

    except Exception as e:
        raise FileReadException(e) from e

    return df


def prepare_raw_data_and_save(input_paths: list[Path], output_dir: Path) -> int:
    """Zwraca liczbe plikow przetworzonych bez bledu."""
    print(f'input_files: {len(input_paths)}')
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    ok_count = 0
    for index, path in enumerate(input_paths, start=1):
        try:
            print(index, '/', len(input_paths))
            df = read_csv(path)
            check_columns(df)
            df['session_index'] = path.stem
            transformed = transform_raw_df(df)
            save_data(transformed, output_dir, path.name)

        except FileReadException as e:
            print(f'ERROR blad odczytu pliku {path} {e} ')
        except WrongColumnsException as e:
            print(f'ERROR nieprawidlowe kolumny w pliku {path} {e}')
        except TransformException as e:
            print(f'ERROR blad transformacji pliku {path} {e}')

        except SaveException as e:
            print(f'ERROR blad zapisu pliku {path} {e} ')

        except Exception as e:
            print(f'ERROR nieznany blad {e}')
        else:
            ok_count += 1
            print(f'OK transformacja {path}')

    print(f'przetworzono {ok_count}/{len(input_paths)} plikow')
    return ok_count


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input_files',
        nargs="*",
        help='Ścieżki surowych danych .csv'
        , default=None
    )
    parser.add_argument(
        "--input_dir",
        help="Katalog z surowymi danymi .csv",
        default=None
    )

    parser.add_argument(
        "--output_dir",
        help="Folder zapisu przygotowanych plików",
        default="prepared",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.input_dir and args.input_files:
        raise ValueError("Podaj albo input_dir albo input_files, nie oba")

    if args.input_dir:
        input_paths = FilesManager.get_csv_paths(args.input_dir)
    elif args.input_files:
        input_paths = [Path(p) for p in args.input_files]
    else:
        raise ValueError("Musisz podać input_dir albo input_files")

    if not input_paths:
        raise ValueError("Nie znaleziono zadnego pliku .csv")

    ok_count = prepare_raw_data_and_save(input_paths, Path(args.output_dir))

    return 0 if ok_count == len(input_paths) else 1


if __name__ == '__main__':
    try:
        exit_code = main()
    except Exception as exc:
        print(f'ERROR {exc}')
        exit_code = 2

    # pauza tylko przy uruchomieniu z konsoli (dwuklik w Windows), nie w skryptach/CI
    if sys.stdin is not None and sys.stdin.isatty():
        input('exit...')

    raise SystemExit(exit_code)
