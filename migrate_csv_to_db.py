"""One-off backfill: load every dataset/*.csv row into the MariaDB weather_readings table.

Usage:
    python migrate_csv_to_db.py [--data-dir dataset]

Safe to re-run: reading_time has a UNIQUE key, so re-imported rows are skipped
via INSERT IGNORE.
"""

import argparse
import glob
import os

import pandas as pd

import weather_db

BATCH_SIZE = 1000


def load_csv_rows(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["timestamp"].astype(str).str.lower() != "timestamp"]

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    for col in ["temp", "humidity", "pressure", "rain_flag"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["temp", "humidity", "pressure", "rain_flag"])

    for _, row in df.iterrows():
        yield (
            row["timestamp"].isoformat(sep=" "),
            float(row["temp"]),
            float(row["humidity"]),
            float(row["pressure"]),
            float(row["rain_flag"]),
            str(row.get("rain")),
        )


def migrate(data_dir):
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return

    weather_db.init_db()

    grand_total_read = 0
    grand_total_inserted = 0

    for csv_file in csv_files:
        batch = []
        file_read = 0
        file_inserted = 0
        for record in load_csv_rows(csv_file):
            batch.append(record)
            file_read += 1
            if len(batch) >= BATCH_SIZE:
                file_inserted += weather_db.insert_readings_bulk(batch)
                batch = []
        if batch:
            file_inserted += weather_db.insert_readings_bulk(batch)

        grand_total_read += file_read
        grand_total_inserted += file_inserted
        print(f"{csv_file}: read {file_read} rows, inserted {file_inserted} new rows")

    print(
        f"Done. Read {grand_total_read} rows total, "
        f"inserted {grand_total_inserted} new rows "
        f"(duplicates skipped via reading_time unique key)."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill CSV history into MariaDB.")
    parser.add_argument(
        "--data-dir",
        default=os.getenv("WEATHER_DATA_DIR", "dataset"),
        help="Directory containing history_*.csv files.",
    )
    args = parser.parse_args()
    migrate(args.data_dir)
