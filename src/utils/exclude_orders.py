# Excluding orders not producing ['analog', 'cover', 'display', 'gyroscope', 'pcb', 'rass1', 'rass2', 'rass3', 'stopper']
# Define the main mapping (allowed parts) as used in unify_and_drop_part_ids
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

# Allowed part_ids (the values in main_mapping)
allowed_ids = set(main_mapping.values())

# Group final_data by order_id and get the set of part_ids for each order
order_parts = final_data.groupby("order_id")["part_id"].apply(set)

# Identify valid orders: orders with part_ids solely from allowed_ids
valid_order_ids = order_parts[
    order_parts.apply(
        lambda parts: parts.issuperset(allowed_ids)
    )  # TODO: issubset or issuperset? Here we filter for the "correct" orders only containing one variant of the product
].index

# Filter final_data (while preserving the original row order) TODO: Exclude or include
final_data_filtered = final_data[final_data["order_id"].isin(valid_order_ids)]

print("Number of valid orders:", final_data_filtered["order_id"].nunique())
