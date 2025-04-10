import pandas as pd


def unify_and_drop_part_ids(real_data: pd.DataFrame) -> pd.DataFrame:
    # Define the main mapping (correct mapping)
    main_mapping = {
        "stopper": 0,
        "cover": 1,
        "rass1": 2,
        "analog": 3,
        "gyroscope": 4,
        "rass2": 5,
        "rass3": 6,
        "pcb": 7,
        "display": 8,
    }

    # Define the wrong mapping (current mapping in the data)
    wrong_mapping = {
        "stopper": 0,
        "robot": 1,
        "gyroscope": 2,
        "cover": 3,
        "pcb": 4,
        "display": 5,
        "analog": 6,
        "rass2": 7,
        "rass3": 8,
        "mainpcb": 9,
        "schield": 10,
        "rass1": 11,
        "frontcover": 12,
        "shield": 13,
        "weather": 14,
        "workpiece": 15,
    }

    # Create reverse mapping from wrong mapping: wrong_id -> category
    wrong_mapping_rev = {v: k for k, v in wrong_mapping.items()}

    # Define a function to update the part_id for a row.
    def update_part_id(row):
        wrong_id = row["part_id"]
        # Look up category using wrong_mapping_rev. If not found, return None.
        category = wrong_mapping_rev.get(wrong_id)
        if category is None:
            return None
        # If the category is in the main mapping, return the corresponding id
        if category in main_mapping:
            return main_mapping[category]
        else:
            # Otherwise, return None to indicate the row should be dropped.
            return None

    # Create a new column with the unified part ids.
    real_data["updated_part_id"] = real_data.apply(update_part_id, axis=1)

    # Drop rows where updated_part_id is None (meaning category not in the main mapping)
    unified_data = real_data.dropna(subset=["updated_part_id"]).copy()

    # Convert updated_part_id to integer
    unified_data["updated_part_id"] = unified_data["updated_part_id"].astype(int)

    # Optionally, replace the original part_id with the unified one and drop the temporary column.
    unified_data["part_id"] = unified_data["updated_part_id"]
    unified_data = unified_data.drop(columns=["updated_part_id"])

    return unified_data


def unify_and_drop_process_types(real_data: pd.DataFrame) -> pd.DataFrame:
    # Main (correct) mapping for process_type.
    main_mapping = {"machine": 0, "feature": 1, "endproduct": 2}

    # Wrong mapping from the data.
    wrong_mapping = {
        "machine": 0,
        "endproduct": 1,
        "feature": 2,
        "transport": 3,
        "test": 4,
    }

    # Create a reverse mapping: wrong_id -> category
    wrong_mapping_rev = {v: k for k, v in wrong_mapping.items()}

    # Function to update the process_type for each row.
    def update_process_type(row):
        wrong_id = row["process_type"]
        category = wrong_mapping_rev.get(wrong_id)
        if category is None or category not in main_mapping:
            # Drop row if the category is not in main mapping (i.e., transport, test)
            return None
        return main_mapping[category]

    # Create a new column with unified process_types.
    real_data["updated_process_type"] = real_data.apply(update_process_type, axis=1)

    # Drop rows without a valid unified process_type.
    unified_data = real_data.dropna(subset=["updated_process_type"]).copy()

    # Convert to integer.
    unified_data["updated_process_type"] = unified_data["updated_process_type"].astype(
        int
    )

    # Replace original process_type with the unified value.
    unified_data["process_type"] = unified_data["updated_process_type"]
    unified_data = unified_data.drop(columns=["updated_process_type"])

    return unified_data


# Define the wrong mapping (used in real data) and create its reverse:
wrong_mapping = {
    "CP-MOBI-WORK-DOCK": 0,
    "CP-F-ASRS20-B": 1,
    "CP-F-ASRS32-P": 2,
    "CP-F-PALROB-B": 3,
    "CP-PICKSORT": 4,
    "CP-AM-CAM": 5,
    "CP-AM-iPICK": 6,
    "CP-F-RASS-1": 7,
    "CP-AM-MEASURE": 8,
    "CP-F-RASS-2": 9,
    "CP-F-RASS-3": 10,
    "CP-AM-FTEST": 11,
    "CP-AM-LABEL": 12,
    "CP-AM-OUT": 13,
    "no resource": 14,
    "CP-F-AASS1": 15,
}
wrong_mapping_rev = {v: k for k, v in wrong_mapping.items()}

# Define the unified mapping based on the main mapping.
# (Note that for mapping from the wrong keys to main IDs, we drop the "_sr name" suffix.)
unified_resource = {
    "CP-F-ASRS32-P": 0,
    "CP-F-RASS-1": 1,
    "CP-AM-MEASURE": 2,
    "CP-F-RASS-2": 3,
    "CP-AM-CAM": 4,
    "CP-F-RASS-3": 5,
    "CP-AM-FTEST": 6,
    "CP-AM-LABEL": 7,
    "CP-AM-OUT": 8,
}


def unify_and_filter_resources(df: pd.DataFrame) -> pd.DataFrame:
    """
    For rows from real data (is_valid==1), convert the resource_id from the wrong mapping to the
    main mapping values. Simulated (is_valid==0) rows should already have correct resource_id values.
    Then drop rows whose resource cannot be unified.
    """

    def update_resource(row):
        # Simulated data (is_valid==0) are assumed to already be using main mapping resource IDs (0-8)
        if row["is_valid"] == 0:
            # Keep the row only if resource_id is in the main mapping (0 through 8).
            if 0 <= row["resource_id"] <= 8:
                return row["resource_id"]
            else:
                return None
        else:
            # For the real data rows: use the wrong mapping to retrieve the resource name,
            # then look it up in the unified_resource mapping.
            wrong_id = row["resource_id"]
            resource_name = wrong_mapping_rev.get(wrong_id)
            if resource_name in unified_resource:
                return unified_resource[resource_name]
            else:
                return None

    df["updated_resource_id"] = df.apply(update_resource, axis=1)
    df = df.dropna(subset=["updated_resource_id"]).copy()
    df["updated_resource_id"] = df["updated_resource_id"].astype(int)
    # Replace the original resource_id with the unified value.
    df["resource_id"] = df["updated_resource_id"]
    df.drop(columns=["updated_resource_id"], inplace=True)
    return df
