import graphviz
import torch
import torchview


def plot(model):
    graphviz.set_jupyter_format("png")
    model_graph = torchview.draw_graph(
        model,
        input_data=torch.randn(1, 19, 10),
        roll=True,
        graph_name="BiLSTM for VVUQ",
        graph_dir="TB",
        save_graph=True,
        expand_nested=True,
    )
    # Schriftart und -größe global setzen
    model_graph.visual_graph.graph_attr.update(
        fontname="Times New Roman", fontsize="14"
    )
    model_graph.visual_graph.node_attr.update(fontname="Times New Roman", fontsize="14")
    model_graph.visual_graph.edge_attr.update(fontname="Times New Roman", fontsize="14")
    return model_graph.visual_graph
