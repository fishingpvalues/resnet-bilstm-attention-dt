import pandas as pd
from collections import Counter
import json

# exclude both for sim and real data part_id of -1
sim_data = sim_data[sim_data["part_id"] != -1]
real_data = real_data[real_data["part_id"] != -1]


def get_order_config(df):
    # Load the part mapping
    with open("output/part_mapping.json", "r") as f:
        part_mapping = json.load(f)

    # Invert the part mapping for easier lookup
    id_to_part = {v: k for k, v in part_mapping.items()}

    # Group part IDs by order ID
    order_configurations = df.groupby("order_id")["part_id"].apply(set).to_dict()

    # Convert part IDs to names
    order_configurations_named = {
        order_id: [id_to_part[part_id] for part_id in part_ids]
        for order_id, part_ids in order_configurations.items()
    }
    return order_configurations_named


def analyze_configurations(config_dict):
    # Convert configurations to tuples for hashability
    configs = [tuple(sorted(config)) for config in config_dict.values()]

    # Count configuration frequencies
    config_counts = Counter(configs)

    # Analyze component presence
    all_components = set(comp for config in configs for comp in config)

    print("Configuration Analysis for:")
    print("\n1. Configuration Frequencies:")
    for config, count in config_counts.most_common():
        print(f"  {list(config)}: {count} times")

    print("\n2. Component Presence:")
    for component in sorted(all_components):
        presence = sum(1 for config in configs if component in config)
        presence_percentage = (presence / len(configs)) * 100
        print(f"  {component}: {presence} times ({presence_percentage:.2f}%)")

    print("\n3. Configuration Complexity:")
    component_counts = [len(config) for config in configs]
    print(f"  Minimum Components: {min(component_counts)}")
    print(f"  Maximum Components: {max(component_counts)}")
    print(f"  Average Components: {sum(component_counts)/len(component_counts):.2f}")


config_sim_data = get_order_config(sim_data)
config_real_data = get_order_config(real_data)

print(analyze_configurations(config_sim_data))
print(analyze_configurations(config_real_data))
