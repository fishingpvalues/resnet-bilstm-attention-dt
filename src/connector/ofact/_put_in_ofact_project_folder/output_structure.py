import pandas as pd
import datetime as dt


class OutputStructure:
    def __init__(
        self,
        process_execution_id: str,
        start_time: str,
        end_time: str,
        part_id: str,
        resource_id: str,
        order_id: str,
        process_id: str,
        event_type: str,
    ):
        self.process_execution_id = process_execution_id  # done
        self.start_time = start_time  # done
        self.end_time = end_time  # done
        self.part_id = part_id  # have?
        self.resource_id = resource_id  # have
        self.order_id = order_id  # have
        self.process_id = process_id  # have
        self.type = event_type  # ?
        self.is_valid = True  # added, default to True

    def to_dict(self) -> dict:
        return {
            "process_execution_id": self.process_execution_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "part_id": self.part_id,
            "resource_id": self.resource_id,
            "order_id": self.order_id,
            "process_id": self.process_id,
            "type": self.type,
            "is_valid": self.is_valid,  # added to dictionary
        }

    @staticmethod
    def create_dataframe() -> pd.DataFrame:
        columns = [
            "process_execution_id",
            "start_time",
            "end_time",
            "duration",  # added new column for duration in seconds
            "part_id",
            "resource_id",
            "order_id",
            "process_id",
            "type",
            "is_valid",  # added new column
        ]
        df = pd.DataFrame(columns=columns)
        df["is_valid"] = True  # default True for is_valid column
        df.set_index("order_id", inplace=True)
        return df

    @staticmethod
    def convert_columns(df: pd.DataFrame) -> pd.DataFrame:
        df["start_time"] = pd.to_datetime(df["start_time"], utc=True)
        df["end_time"] = pd.to_datetime(df["end_time"], utc=True)
        # Calculate duration in seconds as a float
        df["duration"] = (df["end_time"] - df["start_time"]).dt.total_seconds()
        df["process_execution_id"] = df["process_execution_id"].astype(int)
        df["part_id"] = df["part_id"].astype(int)
        df["resource_id"] = df["resource_id"].astype(int)
        df["process_id"] = df["process_id"].astype(int)
        if "is_valid" in df.columns:
            df["is_valid"] = df["is_valid"].astype(bool)
        return df
