import pandas as pd
from datetime import timedelta
from kpi_utils import KPIFactory


def add_kpi_features(final_data: pd.DataFrame) -> pd.DataFrame:
    """
    Enriches final_data with KPI features computed per order.
    For KPIs that do not naturally roll over the process, the overall KPI for the order
    is inserted into every row corresponding to that order.

    New features added:
      - throughput: parts per hour in the order.
      - setup_time_sec: total setup time (in seconds) for the order.
      - lead_time_sec: lead time per (order_id, part_id) as seconds.
      - cycle_time_sec: cycle time per (order_id, part_id) as seconds.

    Assumes final_data has columns including:
      - order_id, part_id, start_time, end_time, process_type, duration

    Returns:
        DataFrame enriched with new KPI columns.
    """
    df = final_data.copy()

    # --- LEAD TIME ---
    # Calculate lead time per (order_id, part_id) from earliest start to latest end.
    lead_df = KPIFactory.calc_lead_time(df)
    # Convert lead_time from timedelta to seconds.
    lead_df["lead_time_sec"] = lead_df["lead_time"].dt.total_seconds()
    lead_df = lead_df[["order_id", "part_id", "lead_time_sec"]]

    # --- CYCLE TIME ---
    cycle_df = KPIFactory.calc_cycle_time(df)
    # If cycle_time is a timedelta, convert to seconds.
    if pd.api.types.is_timedelta64_dtype(cycle_df["cycle_time"]):
        cycle_df["cycle_time_sec"] = cycle_df["cycle_time"].dt.total_seconds()
    else:
        cycle_df["cycle_time_sec"] = cycle_df["cycle_time"]
    cycle_df = cycle_df[["order_id", "part_id", "cycle_time_sec"]]

    # Merge lead time and cycle time back into the data on order_id and part_id.
    df = df.merge(lead_df, on=["order_id", "part_id"], how="left")
    df = df.merge(cycle_df, on=["order_id", "part_id"], how="left")

    # --- OVERALL ORDER KPIs: throughput and setup time ---
    # For each order, calculate throughput and setup_time, then broadcast these values to all rows.
    def compute_order_kpis(group: pd.DataFrame) -> pd.DataFrame:
        # Throughput: parts per hour in this order.
        throughput = KPIFactory.calc_throughput(group, time_unit="H")
        # Setup time: sum of durations for processes with process_type == 1.
        setup_td = KPIFactory.calc_setup_time(group)
        setup_sec = (
            setup_td.total_seconds() if isinstance(setup_td, timedelta) else setup_td
        )
        group = group.copy()
        group["throughput"] = throughput
        group["setup_time_sec"] = setup_sec
        return group

    df = df.groupby("order_id", group_keys=False).apply(compute_order_kpis)

    return df


final_data = add_kpi_features(final_data)
print(final_data.head())
