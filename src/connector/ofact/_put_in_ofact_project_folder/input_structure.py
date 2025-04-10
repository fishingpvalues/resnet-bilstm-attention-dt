from ofact.planning_services.model_generation.persistence import deserialize_state_model

dynm = deserialize_state_model(
    source_file_path=r"D:\ofact-intern\projects\iot_factory\order_sim_iot.pkl",
    persistence_format="pkl",
    dynamics=True,
    deserialization_required=True,
)
print(dynm.get_process_executions_list()[0].start_time)
