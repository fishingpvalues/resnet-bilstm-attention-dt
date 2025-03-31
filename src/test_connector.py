import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

from output_structure import (
    OutputStructure,
)  # assumed to provide a default DataFrame structure


class CSVConnector:
    """Connects the test CSV data to the output structure."""

    def __init__(self, csv_path: Path, output_structure: pd.DataFrame):
        self.csv_path = csv_path
        self.output_structure = output_structure

        # Directory where all mapping files (results and mapping dictionaries) will be saved.
        # Now using data/output instead of the original output folder.
        self.output_dir = Path(r"C:\ofact-intern\projects\iot_factory\val\data\output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> pd.DataFrame:
        """
        Reads the test CSV file, maps its data to the output structure,
        saves mapping dictionaries and the resulting DataFrame to CSV and XLSX files,
        and returns the DataFrame.
        """
        df_in = pd.read_csv(self.csv_path)

        # Generate mapping dictionaries from rows using the 'Description' and other columns.
        part_mapping = self._get_or_create_part_id(df_in)
        type_mapping = self._get_or_create_type(df_in)
        process_mapping = self._get_or_create_process_id(df_in)
        resource_mapping = self._get_or_create_resource_id(df_in)

        # Save mappings to JSON files.
        self._save_mapping(part_mapping, self.output_dir / "part_mapping.json")
        self._save_mapping(type_mapping, self.output_dir / "type_mapping.json")
        self._save_mapping(process_mapping, self.output_dir / "process_mapping.json")
        self._save_mapping(resource_mapping, self.output_dir / "resource_mapping.json")

        # Helper functions to derive part and type ids.
        def get_part_id(desc: str) -> int:
            desc_norm = desc.lower().replace(" ", "")
            for key, pid in part_mapping.items():
                if key in desc_norm:
                    return pid
            return -1

        # Define expert-based type categorization.
        expert_types = {
            "machine": [
                "rass1",
                "rass2",
                "rass3",
                "stopper",
                "boxing",
                "prepare",
                "robot" "assemblystation",
            ],
            "feature": [
                "gyroscope",
                "display",
                "analog",
                "mainpcb",
                "frontcover",
                "pcb",
                "weatherstation",
                "shield",
            ],
            "endproduct": ["deliver", "target", "print label"],
            "test": ["test", "switch", "measure", "check"],
            "transport": ["pick", "store", "place", "read"],
        }

        def get_type_id(desc: str) -> int:
            desc_norm = desc.lower().replace(" ", "")
            for tkey, values in expert_types.items():
                for value in values:
                    if value in desc_norm:
                        return type_mapping.get(tkey, -1)
            return -1

        # Build rows for the new DataFrame.
        rows = []
        for idx, row_in in df_in.iterrows():
            new_row = {}
            new_row["process_execution_id"] = idx
            new_row["order_id"] = str(row_in.get("ONo", ""))
            new_row["start_time"] = pd.to_datetime(row_in.get("Start"))
            new_row["end_time"] = pd.to_datetime(row_in.get("End"))
            new_row["duration"] = (
                new_row["end_time"] - new_row["start_time"]
            ).total_seconds()
            desc = row_in.get("Description", "")

            new_row["part_id"] = get_part_id(desc)
            new_row["process_type"] = get_type_id(desc)
            new_row["process_id"] = process_mapping.get(desc, -1)

            resource = row_in.get("ResourceName", "")
            new_row["resource_id"] = resource_mapping.get(resource, -1)
            new_row["is_valid"] = True

            rows.append(new_row)

        df_out = pd.DataFrame(rows)
        self.output_structure = df_out

        # Save the resulting DataFrame to CSV and XLSX files.
        df_out.to_csv(self.output_dir / "test_no_invalid_entries.csv", index=False)

        return df_out

    def _save_mapping(self, mapping: Dict, path: Path) -> None:
        """Saves a mapping dictionary to a JSON file with pretty formatting."""
        with open(path, "w") as f:
            json.dump(mapping, f, indent=4)

    def _get_or_create_part_id(self, df: pd.DataFrame) -> Dict[str, int]:
        """Creates a part mapping based on unique words found in the 'Description' column."""
        possible_parts = [
            "GYROSCOPE",
            "MAIN PCB",
            "FRONT COVER",
            "RASS1",
            "RASS2",
            "RASS3",
            "display",
            "main PCB",
            "analog",
            "gyroscope",
            "cover",
            "DISPLAY",
            "pcb",
            "ANALOG",
            "stopper",
            "workpiece",
            "weather",
            "shield",
            "schield",
            "robot",
        ]
        normalized_possible = {p.lower().replace(" ", "") for p in possible_parts}
        mapping = {}
        counter = 0
        for desc in df["Description"].dropna():
            desc_norm = desc.lower().replace(" ", "")
            for part in normalized_possible:
                if part in desc_norm and part not in mapping:
                    mapping[part] = counter
                    counter += 1
                    break
        return mapping

    def _get_or_create_type(self, df: pd.DataFrame) -> Dict[str, int]:
        """Creates a type mapping using expert-defined categories based on 'Description'."""
        expert_types = {
            "machine": [
                "rass1",
                "rass2",
                "rass3",
                "stopper",
                "boxing",
                "prepare",
                "robot",
                "assemblystation",
            ],
            "feature": [
                "gyroscope",
                "display",
                "analog",
                "mainpcb",
                "frontcover",
                "pcb",
                "weatherstation",
                "shield",
            ],
            "endproduct": ["deliver", "target", "print label"],
            "test": ["test", "switch", "measure", "check"],
            "transport": ["pick", "store", "place", "read"],
        }
        mapping = {}
        counter = 0
        for desc in df["Description"].dropna():
            desc_norm = desc.lower().replace(" ", "")
            matched_type = None
            for tkey, values in expert_types.items():
                for value in values:
                    if value in desc_norm:
                        matched_type = tkey
                        break
                if matched_type:
                    break
            if matched_type and matched_type not in mapping:
                mapping[matched_type] = counter
                counter += 1
        return mapping

    def _get_or_create_process_id(self, df: pd.DataFrame) -> Dict[str, int]:
        """Assigns an integer ID to each unique process as found in the 'Description'."""
        mapping = {}
        for desc in df["Description"].dropna():
            if desc not in mapping:
                mapping[desc] = len(mapping)
        return mapping

    def _get_or_create_resource_id(self, df: pd.DataFrame) -> Dict[str, int]:
        """Assigns a unique integer to each distinct ResourceName."""
        mapping = {}
        for resource in df["ResourceName"].dropna():
            if resource not in mapping:
                mapping[resource] = len(mapping)
        return mapping


if __name__ == "__main__":
    csv_path = Path(r"C:\ofact-intern\projects\iot_factory\val\data\test.csv")
    output_structure = OutputStructure.create_dataframe()
    connector = CSVConnector(csv_path, output_structure)
    df_out = connector.connect()
