# Create a mask to select only the desired date ranges:
mask = (
    # 20.04.2022 bis einschl. 22.04.2022
    (
        (real_data["start_time"] >= "2022-04-20")
        & (real_data["start_time"] <= "2022-04-22")
    )
    |
    # 02.05.2022 bis einschl. 19.07.2022
    (
        (real_data["start_time"] >= "2022-05-02")
        & (real_data["start_time"] <= "2022-07-19")
    )
    |
    # 02.11.2022 (only this single day)
    (real_data["start_time"].dt.normalize() == pd.Timestamp("2022-11-02"))
    |
    # 11.11.2022
    (real_data["start_time"].dt.normalize() == pd.Timestamp("2022-11-11"))
    |
    # 23.11.2022
    (real_data["start_time"].dt.normalize() == pd.Timestamp("2022-11-23"))
    |
    # März 2023 komplett – from 2023-03-01 to 2023-03-31 (inclusive)
    (
        (real_data["start_time"] >= "2023-03-01")
        & (real_data["start_time"] < "2023-04-01")
    )
)

# Apply the mask to filter the DataFrame.
filtered_real_data = real_data.loc[mask]


# Concatenate the filtered data with the simulated data. Here the real data is at first because of the time horizon.
final_data = pd.concat([filtered_real_data, sim_data]).fillna(0)


# drop process_id > 26
final_data = final_data[final_data["process_id"] <= 26]

# drop part_id = -1
final_data = final_data[final_data["part_id"] != -1]

final_data["sequence_number"] = (
    final_data.sort_values(by=["start_time"]).groupby("process_execution_id").cumcount()
    + 1
)

# Convert start_time and end_time to datetime with UTC
final_data["start_time"] = pd.to_datetime(final_data["start_time"], utc=True)
final_data["end_time"] = pd.to_datetime(final_data["end_time"], utc=True)

# Convert datetime to Unix timestamps (in seconds)
final_data["start_time_unix"] = final_data["start_time"].view("int64") / 10**9
final_data["end_time_unix"] = final_data["end_time"].view("int64") / 10**9

# Create periodic time features
final_data["day_of_week"] = final_data["start_time"].dt.weekday
final_data["hour_of_day"] = final_data["start_time"].dt.hour

final_data["day_of_week_sin"] = np.sin(2 * np.pi * final_data["day_of_week"] / 7)
final_data["day_of_week_cos"] = np.cos(2 * np.pi * final_data["day_of_week"] / 7)
final_data["hour_of_day_sin"] = np.sin(2 * np.pi * final_data["hour_of_day"] / 24)
final_data["hour_of_day_cos"] = np.cos(2 * np.pi * final_data["hour_of_day"] / 24)

# Date feature engineering

# dummy is not a weekday based on end time with function is_not_weekday
final_data["is_not_weekday"] = final_data["end_time"].apply(is_not_weekday).astype(int)

# dummy is break based on end time with function is_break
final_data["is_break"] = final_data["end_time"].apply(is_break).astype(int)


# Sort by the normalized (date-only) start_time, then order_id and sequence_number.
final_data = final_data.sort_values(
    by=["end_time", "order_id", "sequence_number"],
    key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
).reset_index(drop=True)
