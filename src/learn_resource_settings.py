import pandas as pd
import json

# Read the CSV and parse the datetime columns.
real_data = pd.read_csv(
    r"C:\ofact-intern\projects\iot_factory\val\data\real_factorydata.csv",
    parse_dates=["Start", "End"],
)

# Create a mask to select only the desired date ranges:
mask = (
    ((real_data["Start"] >= "2022-04-20") & (real_data["Start"] <= "2022-04-22"))
    | ((real_data["Start"] >= "2022-05-02") & (real_data["Start"] <= "2022-07-19"))
    | (real_data["Start"].dt.normalize() == pd.Timestamp("2022-11-02"))
    | (real_data["Start"].dt.normalize() == pd.Timestamp("2022-11-11"))
    | (real_data["Start"].dt.normalize() == pd.Timestamp("2022-11-23"))
    | ((real_data["Start"] >= "2023-03-01") & (real_data["Start"] < "2023-04-01"))
)

# Apply the mask to filter the DataFrame.
filtered_real_data = real_data.loc[mask]


def get_schedules(df: pd.DataFrame) -> dict:
    """
    For each ResourceID in the process log data, learn the idle (pause) times.
    Assumes the DataFrame has at least the following columns:
      - 'Start': Timestamp when a process begins
      - 'End': Timestamp when a process ends
      - 'ResourceID': Identifier of the resource
    Only records an idle interval as a pause if the duration exceeds 1 hour.
    Returns a dictionary where each key is a ResourceID and its value is a list of
    idle intervals with keys: "idle_start", "idle_end", "idle_duration".
    """
    schedules = {}
    # Group by ResourceID
    for resource_id, group in df.groupby("ResourceID"):
        group = group.sort_values("Start")
        idle_intervals = []
        previous_end = None
        for _, row in group.iterrows():
            if previous_end is None:
                previous_end = row["End"]
                continue
            current_start = row["Start"]
            idle_duration = (current_start - previous_end).total_seconds()
            if current_start > previous_end and idle_duration > 3600:
                idle_intervals.append(
                    {
                        "idle_start": previous_end,
                        "idle_end": current_start,
                        "idle_duration": idle_duration,
                    }
                )
            previous_end = max(previous_end, row["End"])
        schedules[resource_id] = idle_intervals
    return schedules


def write_pause_times_to_excel(
    schedules: dict, resource_names: dict, excel_file_path: str
):
    """
    Writes the computed idle (pause) times to an Excel file with the structure:
      - Division
      - Resource Name
      - Pause Start
      - Pause End
      - Pause Duration
    The resource name is taken from the CSV's ResourceName column (via the resource_names mapping).
    Since no division information exists in the CSV, the Division column is set to "Undefined".
    """
    records = []
    for resource_id, intervals in schedules.items():
        resource_name = resource_names.get(resource_id, f"Resource_{resource_id}")
        division = "Undefined"
        if not intervals:
            records.append(
                {
                    "Division": division,
                    "Resource Name": resource_name,
                    "Pause Start": None,
                    "Pause End": None,
                    "Pause Duration": None,
                }
            )
        else:
            for interval in intervals:
                records.append(
                    {
                        "Division": division,
                        "Resource Name": resource_name,
                        "Pause Start": interval["idle_start"],
                        "Pause End": interval["idle_end"],
                        "Pause Duration": interval["idle_duration"],
                    }
                )
    df_out = pd.DataFrame(
        records,
        columns=[
            "Division",
            "Resource Name",
            "Pause Start",
            "Pause End",
            "Pause Duration",
        ],
    )

    with pd.ExcelWriter(excel_file_path, engine="openpyxl") as writer:
        df_out.to_excel(writer, index=False, sheet_name="PauseTimes")

    print(f"Pause times saved to {excel_file_path}")


if __name__ == "__main__":
    schedules = get_schedules(filtered_real_data)
    resource_names = dict(
        zip(filtered_real_data["ResourceID"], filtered_real_data["ResourceName"])
    )
    output_excel_path = (
        r"c:\ofact-intern\projects\iot_factory\models\resource\settings_checkme.xlsx"
    )
    write_pause_times_to_excel(schedules, resource_names, output_excel_path)
