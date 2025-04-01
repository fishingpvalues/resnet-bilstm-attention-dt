import pandas as pd
import numpy as np


# Assuming the helper functions are defined elsewhere in this module or imported:
def is_not_weekday(ts: pd.Timestamp) -> bool:
    return ts.weekday() >= 5


def is_break(ts: pd.Timestamp) -> bool:
    return ts.hour in range(12, 13)


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and feature engineers the given dataframe.
    The dataframe is expected to have at least the following columns:
    'start_time', 'end_time', 'process_id', 'part_id', 'process_execution_id', and 'order_id'.
    """
    # Apply date range filters.
    mask = (
        # 20.04.2022 bis einschl. 22.04.2022
        ((df["start_time"] >= "2022-04-20") & (df["start_time"] <= "2022-04-22"))
        |
        # 02.05.2022 bis einschl. 19.07.2022
        ((df["start_time"] >= "2022-05-02") & (df["start_time"] <= "2022-07-19"))
        |
        # 02.11.2022 (only this single day)
        (df["start_time"].dt.normalize() == pd.Timestamp("2022-11-02"))
        |
        # 11.11.2022
        (df["start_time"].dt.normalize() == pd.Timestamp("2022-11-11"))
        |
        # 23.11.2022
        (df["start_time"].dt.normalize() == pd.Timestamp("2022-11-23"))
        |
        # März 2023 komplett – from 2023-03-01 to 2023-03-31 (inclusive)
        ((df["start_time"] >= "2023-03-01") & (df["start_time"] < "2023-04-01"))
    )
    filtered = df.loc[mask].copy()

    # Drop unwanted rows.
    filtered = filtered[filtered["process_id"] <= 26]
    filtered = filtered[filtered["part_id"] != -1]

    # Compute sequence_number based on sorted start_time for each process_execution_id.
    filtered["sequence_number"] = (
        filtered.sort_values(by=["start_time"])
        .groupby("process_execution_id")
        .cumcount()
        + 1
    )

    # Convert start_time and end_time to datetime with UTC.
    filtered["start_time"] = pd.to_datetime(filtered["start_time"], utc=True)
    filtered["end_time"] = pd.to_datetime(filtered["end_time"], utc=True)

    # Create Unix timestamp columns.
    filtered["start_time_unix"] = filtered["start_time"].view("int64") / 10**9
    filtered["end_time_unix"] = filtered["end_time"].view("int64") / 10**9

    # Create periodic time features.
    filtered["day_of_week"] = filtered["start_time"].dt.weekday
    filtered["hour_of_day"] = filtered["start_time"].dt.hour
    filtered["day_of_week_sin"] = np.sin(2 * np.pi * filtered["day_of_week"] / 7)
    filtered["day_of_week_cos"] = np.cos(2 * np.pi * filtered["day_of_week"] / 7)
    filtered["hour_of_day_sin"] = np.sin(2 * np.pi * filtered["hour_of_day"] / 24)
    filtered["hour_of_day_cos"] = np.cos(2 * np.pi * filtered["hour_of_day"] / 24)

    # Date feature engineering using the helper functions.
    filtered["is_not_weekday"] = filtered["end_time"].apply(is_not_weekday).astype(int)
    filtered["is_break"] = filtered["end_time"].apply(is_break).astype(int)

    # Sort by normalized end_time, order_id, and sequence_number.
    filtered = filtered.sort_values(
        by=["end_time", "order_id", "sequence_number"],
        key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
    ).reset_index(drop=True)

    return filtered
