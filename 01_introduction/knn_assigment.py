import numpy as np
import matplotlib.pyplot as plt

# Dataset has no header: 8 features followed by the class (0 or 1).
data = np.loadtxt('C:\\Users\\birav\\Desktop\\git\\UiA\\IKT466-1_26H\\01_introduction\\pima-indians-diabetes.csv', delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

# Shuffle datasettet
indices = np.random.default_rng(42).permutation(len(X))

# Split 60% training, 20% validation, 20% testing
train, val, test = np.split(
    indices, [int(0.6 * len(X)), int(0.8 * len(X))]
)

X_train, y_train = X[train], y[train]
X_val, y_val = X[val], y[val]
X_test, y_test = X[test], y[test]

# Standardize using training data only.
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
std[std == 0] = 1
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std        
X_test = (X_test - mean) / std


def predict(point, k):
    distances = np.sqrt(np.sum((X_train - point) ** 2, axis=1)) #euclidean distance
    nearest = np.argsort(distances)[:k] # sorterer etter avstand og tar de k nærmeste
    return int(np.mean(y_train[nearest]) >= 0.5) #denne returnerer 1 hvis gjennomsnittet av de k nærmeste naboene er >= 0.5, ellers returnerer den 0


ks = [1, 3, 5, 7, 9, 11, 15, 21]
accuracies, losses = [], []
for k in ks:

    predictions = []

    for point in X_test:
        result = predict(point, k)
        predictions.append(result)

    predictions = np.array(predictions)
    tp = np.sum((predictions == 1) & (y_test == 1)) #true positive
    tn = np.sum((predictions == 0) & (y_test == 0)) # true negative
    fp = np.sum((predictions == 1) & (y_test == 0)) # false positive
    fn = np.sum((predictions == 0) & (y_test == 1)) # false negative
    accuracy = (tp + tn) / len(y_test) 
    precision = tp / (tp + fp) 
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    mse = np.mean((predictions - y_test) ** 2)
    accuracies.append(accuracy)
    losses.append(mse)
    print(f'K={k:2} | Accuracy={accuracy:.3f} | Precision={precision:.3f} | 'f'Recall={recall:.3f} | F1={f1:.3f} | MSE={mse:.3f}')
    print(f'Confusion matrix [[TN, FP], [FN, TP]]: [[{tn}, {fp}], [{fn}, {tp}]]')

best = ks[int(np.argmax(accuracies))]
print(f'Best K by test accuracy: {best} (exploratory; use validation for final tuning)')
plt.plot(ks, accuracies, marker='o', label='Accuracy')
plt.plot(ks, losses, marker='o', label='Loss (MSE)')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Score')
plt.title('KNN performance on test data')
plt.legend()
plt.grid(True)
plt.savefig('C:\\Users\\birav\\Desktop\\git\\UiA\\IKT466-1_26H\\01_introduction\\knn_graph.png', dpi=150)
plt.show()
