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