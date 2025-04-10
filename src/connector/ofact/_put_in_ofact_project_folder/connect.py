import json
import pandas as pd
from pathlib import Path
from typing import *
from ofact.planning_services.model_generation.persistence import deserialize_state_model
from output_structure import OutputStructure


class Connector:
    """Connects the deserialized dynamic state model with the validation data structure."""

    def __init__(
        self, dynamic_state_model_pth: Path, output_structure: pd.DataFrame
    ) -> None:
        self.dynamic_state_model_pickle = deserialize_state_model(
            source_file_path=dynamic_state_model_pth,
            persistence_format="pkl",
            dynamics=True,
            deserialization_required=True,
        )
        self.output_structure = output_structure

        # Directory where all output files (results and mappings) will be saved.
        self.output_dir = Path(r"C:\ofact-intern\projects\iot_factory\val\simdata")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def connect(self) -> pd.DataFrame:
        """
        Processes the dynamic state model and returns a DataFrame
        with appended process execution attributes.
        Also saves all categorical mapping dictionaries and the DataFrame
        (as CSV and XLSX files) in the output directory.
        """
        # Retrieve process executions and remove those not marked as actual
        process_executions = (
            self.dynamic_state_model_pickle.get_process_executions_list()
        )
        process_executions = self._remove_non_actual_processes(process_executions)

        # Generate mapping dictionaries from the process executions
        part_mapping = self._get_or_create_part_id(process_executions)
        type_mapping = self._get_or_create_type(process_executions)
        process_mapping = self._get_or_create_process_id(process_executions)
        resource_mapping = self._get_or_create_resource_id(process_executions)

        # Save mapping dictionaries to JSON files in the output folder
        self._save_mapping(part_mapping, self.output_dir / "part_mapping.json")
        self._save_mapping(type_mapping, self.output_dir / "type_mapping.json")
        self._save_mapping(process_mapping, self.output_dir / "process_mapping.json")
        self._save_mapping(resource_mapping, self.output_dir / "resource_mapping.json")

        order_ids_list = self._get_order_id(process_executions)
        unique_ids = self._assign_unique_process_execution_id(process_executions)
        start_times, end_times = self._get_times(process_executions)

        # Helper function to get a part ID for an individual process
        def get_part_id(process_name: str) -> int:
            proc_name_norm = process_name.lower().replace(" ", "")
            for key, pid in part_mapping.items():
                if key in proc_name_norm:
                    return pid
            return -1  # Indicates no match

        # Expert-defined dictionary for process type categorization
        typedict = {
            "machine": ["rass1", "rass2", "rass3", "stopper"],
            "feature": ["gyroscope", "display", "analog", "mainpcb", "frontcover"],
            "endproduct": ["deliver"],
            "test": ["test", "switch"],
        }

        def get_type_id(process_name: str) -> int:
            proc_name_norm = process_name.lower().replace(" ", "")
            for tkey, values in typedict.items():
                for value in values:
                    if value in proc_name_norm:
                        return type_mapping.get(tkey, -1)
            return -1

        # Construct rows for DataFrame
        rows = []
        for i, proc in enumerate(process_executions):
            row = {}
            row["process_execution_id"] = unique_ids[i]
            row["order_id"] = order_ids_list[i]
            row["start_time"] = start_times[i]
            row["end_time"] = end_times[i]
            row["part_id"] = get_part_id(proc.process.name)
            row["process_type"] = get_type_id(proc.process.name)
            row["process_id"] = process_mapping.get(proc.process.name, -1)

            # Resolve resource id using resource name or its string representation if name is absent
            resource = proc.main_resource
            resource_key = resource.name if hasattr(resource, "name") else str(resource)
            row["resource_id"] = resource_mapping.get(resource_key, -1)
            row["is_valid"] = 1  # set is_valid to 1 (True)
            rows.append(row)

        # Create DataFrame with the derived information
        df = pd.DataFrame(rows)
        self.output_structure = df

        # Add column for duration in seconds
        df["duration"] = (
            pd.to_datetime(df["end_time"], utc=True)
            - pd.to_datetime(df["start_time"], utc=True)
        ).dt.total_seconds()

        # Save the resulting DataFrame to CSV and XLSX files in the output folder
        df.to_csv(self.output_dir / "train_no_invalid_entries.csv", index=False)
        # df.to_excel(self.output_dir / "output.xlsx", index=False)

        return df

    def _save_mapping(self, mapping: Dict[Any, Any], path: Path) -> None:
        """Saves a given mapping dictionary to a JSON file with pretty-print formatting."""
        with open(path, "w") as f:
            json.dump(mapping, f, indent=4)

    def _remove_non_actual_processes(self, process_executions) -> List:
        """Removes all processes that are not actual.

        Only process executions whose event type contains 'ACTUAL' (case insensitive)
        are kept; those with 'PLANNED' are skipped.
        """
        return [
            process_info
            for process_info in process_executions
            if "ACTUAL" in str(process_info.event_type).upper()
        ]

    def _get_or_create_part_id(self, process_executions) -> Dict[str, int]:
        """Extracts the part ID from the process executions based on domain expertise.
        Compares normalized process names (lowercase with no whitespace) to a set of possible parts.
        If a match is found, assigns a unique integer ID to that part.
        Returns a mapping dictionary with part names as keys and unique integers as values.
        """
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
            "weather",
            "wetter",
            "station",
        ]
        normalized_possible_parts = {
            part.lower().replace(" ", "") for part in possible_parts
        }

        mapping_dict = {}
        counter = 0
        for process_info in process_executions:
            proc_name = process_info.process.name.lower().replace(" ", "")
            matched_part = None
            for part in normalized_possible_parts:
                if part in proc_name:
                    matched_part = part
                    break
            if matched_part and matched_part not in mapping_dict:
                mapping_dict[matched_part] = counter
                counter += 1

        return mapping_dict

    def _get_or_create_type(self, process_executions) -> Dict[str, int]:
        """Assign a type to each process step using expert-defined categories."""
        typedict = {
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
        mapping_dict = {}
        counter = 0
        for process_info in process_executions:
            proc_name = process_info.process.name.lower().replace(" ", "")
            matched_type = None
            for type_key, type_values in typedict.items():
                for value in type_values:
                    if value in proc_name:
                        matched_type = type_key
                        break
                if matched_type:
                    break
            if matched_type and matched_type not in mapping_dict:
                mapping_dict[matched_type] = counter
                counter += 1

        return mapping_dict

    def _get_order_id(self, process_executions) -> List[str]:
        """Extracts the order ID from the process executions."""
        order_ids = []
        for process_info in process_executions:
            try:
                order_ids.append(str(process_info.order.identification))
            except AttributeError:
                order_ids.append("")
        return order_ids

    def _assign_unique_process_execution_id(self, process_executions) -> List[int]:
        """Assigns a unique integer ID to each process execution."""
        return list(range(len(process_executions)))

    def _get_times(self, process_executions) -> Tuple[List[str], List[str]]:
        """Extracts the executed start and end times from the process executions."""
        start_times = [
            process_info.executed_start_time for process_info in process_executions
        ]
        end_times = [
            process_info.executed_end_time for process_info in process_executions
        ]
        return start_times, end_times

    def _get_or_create_process_id(self, process_executions) -> Dict[str, int]:
        """Encodes a categorical process ID with an integer for each unique process description."""
        mapping_dict = {}
        for process_info in process_executions:
            process_name = process_info.process.name
            if process_name not in mapping_dict:
                mapping_dict[process_name] = len(mapping_dict)
        return mapping_dict

    def _get_or_create_resource_id(self, process_executions) -> Dict[str, int]:
        """
        Scans all process executions and assigns a unique integer ID for each unique main resource.
        The uniqueness is determined by the resource name; if not available, the string representation is used.
        """
        mapping = {}
        counter = 0
        for process_info in process_executions:
            resource = process_info.main_resource
            resource_key = resource.name if hasattr(resource, "name") else str(resource)
            if resource_key not in mapping:
                mapping[resource_key] = counter
                counter += 1
        return mapping


if __name__ == "__main__":
    # Create an empty DataFrame using the OutputStructure class
    df = OutputStructure.create_dataframe()

    # Instantiate the Connector class with the DataFrame and dynamic state model path
    conn = Connector(
        Path(r"C:\ofact-intern\projects\iot_factory\order_sim_iot.pkl"), df
    )
    result_df = conn.connect()
