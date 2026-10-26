from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


###################part 1
data = pd.read_csv(Path(__file__).with_name('titanic.csv'))

data = data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']].copy()

data['Age'] = pd.cut(
    data['Age'],
    bins=[0, 12, 18, 35, 60, np.inf],
    labels=['Child', 'Teen', 'Young', 'Adult', 'Senior']
)

data['Fare'] = pd.cut(
    data['Fare'],
    bins=[-1, 10, 30, 100, np.inf],
    labels=['Low', 'Medium', 'High', 'Very High']
)

# 4. Fjern manglende verdier
data = data.dropna()


def calculate_entropy(data):

    counts = data['Survived'].value_counts()
    total = len(data)

    probabilities = counts / total

    entropy = -sum(probabilities * np.log2(probabilities))
    return entropy


def information_gain(data, feature):

    total_entropy = calculate_entropy(data)

    entropy_after_split = 0

    for value in data[feature].unique():
        subset = data[data[feature] == value]
        weight = len(subset) / len(data)
        entropy_after_split += weight * calculate_entropy(subset)


    info_gain = total_entropy - entropy_after_split
    return info_gain

def highest_information_gain(data):
    best_gain = -1
    best_column = None
    features = data.columns.drop('Survived')
    for feature in features:
        gain = information_gain(data, feature)

        if gain > best_gain:
            best_gain = gain
            best_column = feature


    return best_column

def build_tree(data):
    if data['Survived'].nunique() == 1:
        return int(data['Survived'].iloc[0])

    if len(data.columns) == 1:
        return int(data['Survived'].mode()[0])
        
    best_feature = highest_information_gain(data) #best kolonne å splitte på 

    if information_gain(data, best_feature) <= 0:
        return int(data['Survived'].mode()[0])

    tree = {best_feature: {}}


    for value in data[best_feature].unique():

        subset = data[data[best_feature] == value]

        subset = subset.drop(columns=[best_feature])

        tree[best_feature][value] = build_tree(subset)

    return tree


def predict(tree, passenger):

    if not isinstance(tree, dict):
        return tree

    feature = next(iter(tree))

    value = passenger[feature]

    if value not in tree[feature]:
        return None #incase det ligger en ukjent verdi her 

    return predict(tree[feature][value], passenger)

train_data, test_data = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

tree = build_tree(train_data)

predictions = []

for _, passenger in test_data.iterrows():

    prediction = predict(tree, passenger)

    if prediction is None:
        prediction = int(train_data['Survived'].mode()[0])

    predictions.append(prediction)


# Lag en kopi av testdatasettet
results = test_data.copy()

# Legg til prediksjonene
results['Prediction'] = predictions

# Konverter 0 og 1 til tekst
results['Prediction'] = results['Prediction'].map({
    0: 'Not Survived',
    1: 'Survived'
})

# Vis resultatene
print(results[['Sex', 'Age', 'Pclass', 'Survived', 'Prediction']].head(10))

# Beregn nøyaktigheten
accuracy = accuracy_score(test_data['Survived'], predictions)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

###################part 1

###################part 2
def build_random_forest(data, n_trees=10): ## mekker 10 en random trees

    forest = []

    for i in range(n_trees):

        sample = data.sample(
            n=len(data),
            replace=True,
            random_state=42 + i
        )

        tree = build_tree(sample)

        forest.append(tree)

    return forest

def random_forest_predict(forest, passenger, default_prediction): ##majority voting

    votes = []

    for tree in forest:

        prediction = predict(tree, passenger)

        if prediction is None:
            prediction = default_prediction

        votes.append(prediction)

    return max(set(votes), key=votes.count)

forest = build_random_forest(train_data, n_trees=11) #bruker train_data for å bygge 11 random trees

default_prediction = int(train_data['Survived'].mode()[0])

rf_predictions = []

for _, passenger in test_data.iterrows():

    prediction = random_forest_predict(
        forest,
        passenger,
        default_prediction
    )

    rf_predictions.append(prediction)

id3_accuracy = accuracy_score(
    test_data['Survived'],
    predictions
)

rf_accuracy = accuracy_score(
    test_data['Survived'],
    rf_predictions
)

print(f"ID3 Accuracy: {id3_accuracy * 100:.2f}%")
print(f"Random Forest Accuracy: {rf_accuracy * 100:.2f}%")

results = test_data.copy()

results['Random Forest Prediction'] = [
    'Survived' if p == 1 else 'Not Survived'
    for p in rf_predictions
]

print(results[['Sex', 'Age', 'Pclass', 'Survived',
               'Random Forest Prediction']].head(10))


### oppg 3
################### part 3

c45_data = pd.read_csv(Path(__file__).with_name('titanic.csv'))

c45_data = c45_data[
    ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
].copy()

# konvertering av tekst til tall
c45_data['Sex'] = c45_data['Sex'].map({'male': 0, 'female': 1})
c45_data['Embarked'] = c45_data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Fyller inn manglende verdier
for column in c45_data.columns:
    c45_data[column] = c45_data[column].fillna(
        c45_data[column].median()
    )

c45_train, c45_test = train_test_split(
    c45_data,
    test_size=0.2,
    random_state=42
)

def gain_ratio(data, feature, threshold):

    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if len(left) == 0 or len(right) == 0:
        return 0

    total = len(data)

    left_weight = len(left) / total
    right_weight = len(right) / total

    entropy_before = calculate_entropy(data)

    entropy_after = (
        left_weight * calculate_entropy(left)
        + right_weight * calculate_entropy(right)
    )

    gain = entropy_before - entropy_after

    split_info = -(
        left_weight * np.log2(left_weight)
        + right_weight * np.log2(right_weight)
    )

    return gain / split_info

def best_c45_split(data):

    best_gain = 0
    best_feature = None
    best_threshold = None

    for feature in data.columns.drop('Survived'):

        values = sorted(data[feature].unique())

        for i in range(len(values) - 1):

            threshold = (values[i] + values[i + 1]) / 2

            gain = gain_ratio(data, feature, threshold)

            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold

    return best_feature, best_threshold

def build_c45_tree(data): #dette bygger C45 treet

    majority = int(data['Survived'].mode()[0])

    if data['Survived'].nunique() == 1:
        return majority

    feature, threshold = best_c45_split(data)

    if feature is None:
        return majority

    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    tree = {
        'feature': feature,
        'threshold': threshold,
        'majority': majority,
        'left': build_c45_tree(left),
        'right': build_c45_tree(right)
    }

    return tree

def predict_c45(tree, passenger):

    if not isinstance(tree, dict):
        return tree

    feature = tree['feature']
    threshold = tree['threshold']

    if passenger[feature] <= threshold:
        return predict_c45(tree['left'], passenger)

    return predict_c45(tree['right'], passenger)

def prune_tree(tree, validation):

    if not isinstance(tree, dict) or len(validation) == 0:
        return tree

    feature = tree['feature']
    threshold = tree['threshold']

    left = validation[validation[feature] <= threshold]
    right = validation[validation[feature] > threshold]

    tree['left'] = prune_tree(tree['left'], left)
    tree['right'] = prune_tree(tree['right'], right)

    actual = validation['Survived']

    predictions = [
        predict_c45(tree, passenger)
        for _, passenger in validation.iterrows()
    ]

    tree_accuracy = accuracy_score(actual, predictions)

    majority_accuracy = accuracy_score(
        actual,
        [tree['majority']] * len(validation)
    )

    if majority_accuracy >= tree_accuracy:
        return tree['majority']

    return tree

# Del treningsdata i trening og validering
train_part, validation_part = train_test_split(
    c45_train,
    test_size=0.2,
    random_state=42
)

# Bygg treet
c45_tree = build_c45_tree(train_part)

# Fjern unødvendige grener
c45_tree = prune_tree(c45_tree, validation_part)

# Prediksjoner
c45_predictions = []

for _, passenger in c45_test.iterrows():
    prediction = predict_c45(c45_tree, passenger)
    c45_predictions.append(prediction)

# Accuracy
c45_accuracy = accuracy_score(
    c45_test['Survived'],
    c45_predictions
)

print(f"C4.5 Accuracy: {c45_accuracy * 100:.2f}%")

