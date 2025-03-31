# Train Test Split
final_data = final_data.sort_values(
    by=["end_time", "order_id", "sequence_number"],
    key=lambda col: col.dt.normalize() if col.name == "end_time" else col,
).reset_index(drop=True)

X = final_data.drop(
    columns=[
        "is_valid",
        "start_time",
        "end_time",
        "order_id",
        "start_time_unix",
        "end_time_unix",
    ]
)


y = final_data["is_valid"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

from sklearn.tree import DecisionTreeClassifier

plt.figure(figsize=(20, 10), dpi=300)


dt = DecisionTreeClassifier(random_state=683, max_depth=5)


dt.fit(X_train, y_train)


y_pred = dt.predict(X_test)


y_proba = dt.predict_proba(X_test)[:, 1]


generate_report(y_test, y_pred, y_proba)


import graphviz
from sklearn.tree import export_graphviz

# Export the decision tree to a Graphviz dot file
dot_data = export_graphviz(
    dt,
    out_file=None,
    feature_names=X_train.columns,
    class_names=["0", "1"],
    filled=True,
    rounded=True,
    special_characters=True,
)

# Render the graph using Graphviz
graph = graphviz.Source(dot_data)
graph.render("decision_tree")  # Saves to a file
graph.view()  # Opens the tree in a viewer
