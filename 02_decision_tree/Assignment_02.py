import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. Last inn Titanic-data
from pathlib import Path
fil = Path(__file__).with_name('titanic.csv')
data = pd.read_csv(fil)
kolonner = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
X = data[kolonner].copy()
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X['Embarked'] = X['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
y = data['Survived'].to_numpy(dtype=int)

# Del først, og fyll deretter manglende verdier med median fra treningsdata.
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
medianer = X_train_df.median()
X_train = X_train_df.fillna(medianer).to_numpy(dtype=float)
X_test = X_test_df.fillna(medianer).to_numpy(dtype=float)

# ID3 bruker kategorier for Age og Fare.
def make_categories(X):
    X = X.copy()
    X[:, 2] = np.digitize(X[:, 2], [18, 40])
    X[:, 5] = np.digitize(X[:, 5], [10, 30])
    return X

# 2. Entropi: hvor blandet er overlevelsen?
def entropy(y):
    if len(y) == 0:
        return 0
    result = 0
    for count in [sum(y == 0), sum(y == 1)]:
        if count > 0:
            probability = count / len(y)
            result -= probability * np.log2(probability)
    return result

# 3. Finn beste splitt: information gain for ID3, gain ratio for C4.5.
def best_split(X, y, method, available):
    best_score = 0
    best_column = None
    best_threshold = None
    best_groups = None

    for column in available:
        values = np.unique(X[:, column])
        if method == 'ID3':
            groups = []
            for value in values:
                groups.append(X[:, column] == value)
            choices = [(None, groups)]
        else:
            choices = []
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left = X[:, column] <= threshold
                right = X[:, column] > threshold
                choices.append((threshold, [left, right]))

        for threshold, groups in choices:
            gain = entropy(y)
            split_info = 0
            valid = True
            for group in groups:
                fraction = sum(group) / len(y)
                if fraction == 0:
                    valid = False
                    break
                gain -= fraction * entropy(y[group])
                split_info -= fraction * np.log2(fraction)
            if not valid or split_info == 0:
                continue
            score = gain
            if method == 'C4.5':
                score = gain / split_info
            if score > best_score:
                best_score = score
                best_column = column
                best_threshold = threshold
                best_groups = groups
    return best_column, best_threshold, best_groups

# 4. Bygg et beslutningstre.
def build_tree(X, y, method='ID3', depth=0, available=None):
    majority = 1 if sum(y == 1) > sum(y == 0) else 0
    if available is None:
        available = list(range(X.shape[1]))
    if len(set(y)) == 1 or depth >= 7 or len(y) < 5 or not available:
        return majority

    column, threshold, groups = best_split(X, y, method, available)
    if column is None:
        return majority

    remaining = available
    if method == 'ID3':
        remaining = []
        for feature in available:
            if feature != column:
                remaining.append(feature)

    children = []
    for group in groups:
        child = build_tree(X[group], y[group], method, depth + 1, remaining)
        children.append(child)

    values = None
    if method == 'ID3':
        values = list(np.unique(X[:, column]))
    return {
        'column': column,
        'threshold': threshold,
        'values': values,
        'children': children,
        'majority': majority
    }



# 5. Prediker én eller flere passasjerer.
def predict_one(tree, passenger):
    while isinstance(tree, dict):
        column = tree['column']
        if tree['values'] is None:
            branch = 0 if passenger[column] <= tree['threshold'] else 1
        else:
            value = passenger[column]
            if value not in tree['values']:
                return tree['majority']
            branch = tree['values'].index(value)
        tree = tree['children'][branch]
    return tree


def predict(tree, X):
    predictions = []
    for passenger in X:
        predictions.append(predict_one(tree, passenger))
    return np.array(predictions)

# 6. Post-pruning: fjern grener som ikke hjelper på valideringsdata.
def prune(tree, X, y):
    if not isinstance(tree, dict) or len(y) == 0:
        return tree
    for index in range(len(tree['children'])):
        column = tree['column']
        if tree['values'] is not None:
            mask = X[:, column] == tree['values'][index]
        elif index == 0:
            mask = X[:, column] <= tree['threshold']
        else:
            mask = X[:, column] > tree['threshold']
        child = tree['children'][index]
        tree['children'][index] = prune(child, X[mask], y[mask])

    original_accuracy = np.mean(predict(tree, X) == y)
    simple_accuracy = np.mean(y == tree['majority'])
    if simple_accuracy >= original_accuracy:
        return tree['majority']
    return tree

# 7. Random Forest: lag flere trær på tilfeldige utvalg.
def random_forest(X, y, method, count=7):
    trees = []
    for i in range(count):
        random = np.random.default_rng(42 + i)
        sample = random.choice(len(y), size=len(y), replace=True)
        tree = build_tree(X[sample], y[sample], method)
        trees.append(tree)
    return trees

def vote(trees, X):
    predictions = []
    for passenger in X:
        votes = []
        for tree in trees:
            votes.append(predict_one(tree, passenger))
        if votes.count(1) > votes.count(0):
            predictions.append(1)
        else:
            predictions.append(0)
    return np.array(predictions)

# DEL 1: ID3
print('\n========== DEL 1: ID3 ==========')
X_id3_train = make_categories(X_train)
X_id3_test = make_categories(X_test)
id3 = build_tree(X_id3_train, y_train, 'ID3')
id3_predictions = predict(id3, X_id3_test)
print('Accuracy:', accuracy_score(y_test, id3_predictions))
print('First 10 predictions:', id3_predictions[:10])

# DEL 2: Random Forest ID3
print('\n========== DEL 2: RANDOM FOREST ID3 ==========')
rf_id3 = random_forest(X_id3_train, y_train, 'ID3')
rf_id3_predictions = vote(rf_id3, X_id3_test)
print('Accuracy:', accuracy_score(y_test, rf_id3_predictions))
print('First 10 predictions:', rf_id3_predictions[:10])

# DEL 3: C4.5
print('\n========== DEL 3: C4.5 ==========')
X_fit, X_valid, y_fit, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
c45 = build_tree(X_fit, y_fit, 'C4.5')
c45 = prune(c45, X_valid, y_valid)
c45_predictions = predict(c45, X_test)
print('Accuracy:', accuracy_score(y_test, c45_predictions))
print('First 10 predictions:', c45_predictions[:10])

# DEL 4: Random Forest C4.5
print('\n========== DEL 4: RANDOM FOREST C4.5 ==========')
rf_c45 = random_forest(X_fit, y_fit, 'C4.5')
pruned_forest = []
for tree in rf_c45:
    pruned_forest.append(prune(tree, X_valid, y_valid))
rf_c45 = pruned_forest
rf_c45_predictions = vote(rf_c45, X_test)
print('Accuracy:', accuracy_score(y_test, rf_c45_predictions))
print('First 10 predictions:', rf_c45_predictions[:10])

# SLUTTVURDERING
print('\n========== SLUTTVURDERING ==========')
results = {
    'ID3': id3_predictions,
    'RF-ID3': rf_id3_predictions,
    'C4.5': c45_predictions,
    'RF-C4.5': rf_c45_predictions
}
print('Model       Accuracy  Precision  Recall  F1')
for name, predictions in results.items():
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)
    print(f'{name:10}  {accuracy:.3f}     {precision:.3f}      {recall:.3f}  {f1:.3f}')

