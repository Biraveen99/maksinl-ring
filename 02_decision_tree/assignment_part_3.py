
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ==================== DATA ====================

data = pd.read_csv(Path(__file__).with_name("titanic.csv"))

data = data[
    ["Survived", "Pclass", "Sex", "Age", "Fare", "Embarked"]
].copy()

# Del først opp dataene.
train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42,
    stratify=data["Survived"]
)

# Eget valideringssett for post-pruning.
train_data, validation_data = train_test_split(
    train_data,
    test_size=0.2,
    random_state=42,
    stratify=train_data["Survived"]
)

# Fyll inn manglende verdier basert på treningsdata.
for feature in ["Age", "Fare"]:
    median = train_data[feature].median()

    for dataset in [train_data, validation_data, test_data]:
        dataset[feature] = dataset[feature].fillna(median)

for feature in ["Sex", "Embarked", "Pclass"]:
    most_common = train_data[feature].mode()[0]

    for dataset in [train_data, validation_data, test_data]:
        dataset[feature] = dataset[feature].fillna(most_common)


# ==================== ENTROPY ====================

def calculate_entropy(data):
    counts = data["Survived"].value_counts()
    probabilities = counts / len(data)

    return -sum(probabilities * np.log2(probabilities))


# ==================== GAIN RATIO ====================

def calculate_gain_ratio(data, feature, threshold=None):

    parent_entropy = calculate_entropy(data)

    if threshold is not None:
        subsets = [
            data[data[feature] <= threshold],
            data[data[feature] > threshold]
        ]
    else:
        subsets = [
            data[data[feature] == value]
            for value in data[feature].unique()
        ]

    entropy_after_split = 0
    split_information = 0

    for subset in subsets:
        if len(subset) == 0:
            continue

        weight = len(subset) / len(data)

        entropy_after_split += weight * calculate_entropy(subset)
        split_information -= weight * np.log2(weight)

    information_gain = parent_entropy - entropy_after_split

    if split_information == 0:
        return 0

    return information_gain / split_information


# ==================== BEST SPLIT ====================

def find_best_split(data):

    best_feature = None
    best_threshold = None
    best_ratio = 0

    for feature in data.columns.drop("Survived"):

        if feature in ["Age", "Fare"]:
            values = sorted(data[feature].unique())

            thresholds = [
                (values[i] + values[i + 1]) / 2
                for i in range(len(values) - 1)
            ]

            for threshold in thresholds:
                ratio = calculate_gain_ratio(
                    data, feature, threshold
                )

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_feature = feature
                    best_threshold = threshold

        else:
            ratio = calculate_gain_ratio(data, feature)

            if ratio > best_ratio:
                best_ratio = ratio
                best_feature = feature
                best_threshold = None

    return best_feature, best_threshold, best_ratio


# ==================== BUILD C4.5 TREE ====================

def build_tree(data, depth=0, max_depth=10, min_samples=5):

    majority = int(data["Survived"].mode()[0])

    node = {
        "prediction": majority,
        "feature": None,
        "threshold": None,
        "branches": {}
    }

    if data["Survived"].nunique() == 1:
        return node

    if depth >= max_depth or len(data) < min_samples:
        return node

    feature, threshold, ratio = find_best_split(data)

    if feature is None or ratio <= 0:
        return node

    node["feature"] = feature
    node["threshold"] = threshold

    if threshold is not None:
        left = data[data[feature] <= threshold]
        right = data[data[feature] > threshold]

        node["branches"]["left"] = build_tree(
            left, depth + 1, max_depth, min_samples
        )

        node["branches"]["right"] = build_tree(
            right, depth + 1, max_depth, min_samples
        )

    else:
        for value in data[feature].unique():
            subset = data[data[feature] == value].drop(
                columns=[feature]
            )

            node["branches"][value] = build_tree(
                subset, depth + 1, max_depth, min_samples
            )

    return node


# ==================== PREDICTION ====================

def predict(tree, passenger):

    if tree["feature"] is None:
        return tree["prediction"]

    feature = tree["feature"]
    threshold = tree["threshold"]

    if threshold is not None:
        branch = (
            "left"
            if passenger[feature] <= threshold
            else "right"
        )
    else:
        branch = passenger[feature]

    if branch not in tree["branches"]:
        return tree["prediction"]

    return predict(tree["branches"][branch], passenger)


# ==================== POST-PRUNING ====================

def prune_tree(tree, validation_data):

    if tree["feature"] is None or validation_data.empty:
        return tree

    feature = tree["feature"]
    threshold = tree["threshold"]

    for branch, child in tree["branches"].items():

        if threshold is not None:
            if branch == "left":
                subset = validation_data[
                    validation_data[feature] <= threshold
                ]
            else:
                subset = validation_data[
                    validation_data[feature] > threshold
                ]
        else:
            subset = validation_data[
                validation_data[feature] == branch
            ]

        prune_tree(child, subset)

    # Accuracy med eksisterende subtree.
    current_predictions = [
        predict(tree, row)
        for _, row in validation_data.iterrows()
    ]

    current_accuracy = accuracy_score(
        validation_data["Survived"],
        current_predictions
    )

    # Accuracy dersom noden erstattes med et blad.
    leaf_predictions = [
        tree["prediction"]
    ] * len(validation_data)

    leaf_accuracy = accuracy_score(
        validation_data["Survived"],
        leaf_predictions
    )

    # Fjern subtree hvis bladet er like bra eller bedre.
    if leaf_accuracy >= current_accuracy:
        tree["feature"] = None
        tree["threshold"] = None
        tree["branches"] = {}

    return tree


# ==================== TRAIN & EVALUATE ====================

tree = build_tree(train_data)

before_pruning = [
    predict(tree, passenger)
    for _, passenger in test_data.iterrows()
]

# Beskjær kun ved hjelp av valideringsdata.
tree = prune_tree(tree, validation_data)

predictions = [
    predict(tree, passenger)
    for _, passenger in test_data.iterrows()
]

results = test_data.copy()

results["Prediction"] = [
    "Survived" if prediction == 1 else "Not Survived"
    for prediction in predictions
]

print(
    results[
        ["Sex", "Age", "Pclass", "Survived", "Prediction"]
    ].head(10)
)

accuracy_before = accuracy_score(
    test_data["Survived"], before_pruning
)

accuracy_after = accuracy_score(
    test_data["Survived"], predictions
)

print(f"\nC4.5 Accuracy before pruning: {accuracy_before * 100:.2f}%")
print(f"C4.5 Accuracy after pruning:  {accuracy_after * 100:.2f}%")
