import pandas as pd
from datetime import timedelta
from typing import Dict, Tuple, List


class KPIFactory:
    """Class for calculating KPIs for an IoT factory.
    The data structure is based on an Object-Centric Event Log (OCEL).
    """

    @staticmethod
    def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
        """Ensure that the DataFrame contains the required columns."""
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")

    @staticmethod
    def calc_overall_time_window(df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Calculate the overall time window of the data."""
        KPIFactory.validate_columns(df, ["start_time", "end_time"])
        t_start: pd.Timestamp = df["start_time"].min()
        t_end: pd.Timestamp = df["end_time"].max()
        return t_start, t_end

    @staticmethod
    def calc_throughput(df: pd.DataFrame, time_unit: str = "H") -> float:
        """
        Calculate throughput (parts per time unit).
        Only counts unique (order_id, part_id) pairs with part_id >= 0.
        time_unit : "H" for hours, "T" for minutes, otherwise seconds.
        """
        KPIFactory.validate_columns(
            df, ["order_id", "part_id", "start_time", "end_time"]
        )
        # Count only good parts (part_id >= 0)
        good_parts_df = df[df["part_id"] >= 0]
        unique_parts = good_parts_df.drop_duplicates(subset=["order_id", "part_id"])
        count_parts: int = len(unique_parts)

        t_start, t_end = KPIFactory.calc_overall_time_window(df)
        total_seconds: float = (t_end - t_start).total_seconds()

        if time_unit == "H":
            time_duration = total_seconds / 3600.0
        elif time_unit == "T":
            time_duration = total_seconds / 60.0
        else:
            time_duration = total_seconds

        throughput: float = count_parts / time_duration if time_duration > 0 else 0.0
        return throughput

    @staticmethod
    def calc_lead_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the lead time per part as difference between earliest start and latest end.
        Returns a DataFrame with columns: order_id, part_id, lead_time.
        """
        KPIFactory.validate_columns(
            df, ["order_id", "part_id", "start_time", "end_time"]
        )
        good_parts_df = df[df["part_id"] >= 0].copy()
        lead_times: pd.DataFrame = (
            good_parts_df.groupby(["order_id", "part_id"])
            .agg(start_min=("start_time", "min"), end_max=("end_time", "max"))
            .reset_index()
        )
        lead_times["lead_time"] = lead_times["end_max"] - lead_times["start_min"]
        return lead_times[["order_id", "part_id", "lead_time"]]

    @staticmethod
    def calc_cycle_time(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cycle time per part as the sum of active production times (process_type == 0).
        Returns a DataFrame with columns: order_id, part_id, cycle_time.
        """
        KPIFactory.validate_columns(
            df, ["order_id", "part_id", "process_type", "duration"]
        )
        prod_df = df[df["process_type"] == 0].copy()
        cycle_times: pd.DataFrame = (
            prod_df.groupby(["order_id", "part_id"])
            .agg(cycle_time=("duration", "sum"))
            .reset_index()
        )
        return cycle_times

    @staticmethod
    def calc_setup_time(df: pd.DataFrame) -> timedelta:
        """
        Calculate the total setup time as the sum of durations of all processes with process_type == 1.
        """
        KPIFactory.validate_columns(df, ["process_type", "duration"])
        setup_series = df[df["process_type"] == 1]["duration"]
        if setup_series.empty:
            return timedelta(0)
        if not isinstance(setup_series.iloc[0], timedelta):
            total_setup = timedelta(seconds=setup_series.sum())
        else:
            total_setup = setup_series.sum()
        return total_setup
