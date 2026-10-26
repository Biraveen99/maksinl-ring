# IKT466 - Assignment 06: Backpropagation
# Enkelt nevralt nettverk fra scratch (uten PyTorch/TensorFlow).
# Bygger videre på forward pass fra Assignment 05.

from pathlib import Path
import numpy as np
import pandas as pd


# DEL 1: LAST INN OG NORMALISER DATA
fil = Path(__file__).with_name("ecoli.csv")
data = pd.read_csv(fil)
data = data[data["SITE"].isin(["cp", "im"])].copy()

# Første kolonne er protein-ID, siste kolonne er klassen.
X = data.iloc[:, 1:-1].to_numpy(dtype=float)
y = data["SITE"].map({"cp": 0, "im": 1}).to_numpy(dtype=int)

# Min-max-normalisering til verdier mellom 0 og 1.
minste = X.min(axis=0)
forskjell = X.max(axis=0) - minste
forskjell[forskjell == 0] = 1
X = (X - minste) / forskjell

print("========== DEL 1: DATA ==========")
print("Antall proteiner:", len(X))
print("Antall egenskaper:", X.shape[1])
print("Klasser: cp = 0, im = 1")


# DEL 2: AKTIVERINGSFUNKSJONER
# Sigmoid brukes i output layer og gir et tall mellom 0 og 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# ReLU brukes i hidden layer.
def relu(x):
    return np.maximum(0, x)


# Derivert av ReLU: 1 for positive tall, ellers 0.
def relu_derivative(x):
    return (x > 0).astype(float)


# DEL 3: BINARY CROSS-ENTROPY (BCE)
def binary_cross_entropy(fasit, prediction):
    # Unngår log(0).
    prediction = np.clip(prediction, 0.0000001, 0.9999999)
    return -(fasit * np.log(prediction) + (1 - fasit) * np.log(1 - prediction))


print("\n========== DEL 3: TEST AV BCE ==========")
print("Riktig prediksjon (fasit 1, prediction 0.9):", round(float(binary_cross_entropy(1, 0.9)), 4))
print("Feil prediksjon (fasit 1, prediction 0.1):", round(float(binary_cross_entropy(1, 0.1)), 4))


# DEL 4: INITIALISER VEKTENE TILFELDIG
np.random.seed(42)  # Samme startvekter ved hver kjøring.
antall_inputs = X.shape[1]
antall_hidden = 3

hidden_weights = np.random.randn(antall_inputs, antall_hidden) * 0.5
hidden_biases = np.zeros(antall_hidden)
output_weights = np.random.randn(antall_hidden) * 0.5
output_bias = 0.0


# DEL 5: FORWARD PASS
# Vi regner først ut hidden layer og deretter output layer.
def forward_pass(inputs):
    hidden_total = np.dot(inputs, hidden_weights) + hidden_biases
    hidden_output = relu(hidden_total)
    output_total = np.dot(hidden_output, output_weights) + output_bias
    prediction = sigmoid(output_total)
    return hidden_total, hidden_output, prediction


# DEL 6: BACKPROPAGATION OG SGD
# Bruker chain rule for å finne hvordan vektene skal endres.
def train_one_protein(inputs, fasit, learning_rate):
    global hidden_weights, hidden_biases, output_weights, output_bias

    # 1. Forward pass
    hidden_total, hidden_output, prediction = forward_pass(inputs)
    loss = binary_cross_entropy(fasit, prediction)

    # 2. Feilen i output layer (BCE + sigmoid gir prediction - fasit).
    output_error = prediction - fasit

    # 3. Send feilen bakover til hidden layer (chain rule).
    hidden_error = output_error * output_weights * relu_derivative(hidden_total)

    # 4. Regn ut gradienter FØR vi endrer noen vekter.
    output_weight_gradient = hidden_output * output_error
    hidden_weight_gradient = np.outer(inputs, hidden_error)

    # 5. Oppdater vekter og bias med stochastic gradient descent (SGD).
    output_weights -= learning_rate * output_weight_gradient
    output_bias -= learning_rate * output_error
    hidden_weights -= learning_rate * hidden_weight_gradient
    hidden_biases -= learning_rate * hidden_error

    return loss


# DEL 7: TRENING
learning_rate = 0.05
epochs = 1000

print("\n========== DEL 7: TRENING ==========")
for epoch in range(1, epochs + 1):
    # SGD: bland rekkefølgen og oppdater etter hvert protein.
    rekkefolge = np.random.permutation(len(X))
    for indeks in rekkefolge:
        train_one_protein(X[indeks], y[indeks], learning_rate)

    # Beregn loss med de oppdaterte vektene etter hver 100. epoch.
    if epoch % 100 == 0 or epoch == 1:
        total_loss = 0
        for i in range(len(X)):
            _, _, prediction = forward_pass(X[i])
            total_loss += binary_cross_entropy(y[i], prediction)
        average_loss = total_loss / len(X)
        print(f"Epoch {epoch:4d} - Loss: {average_loss:.4f}")


# DEL 8: EVALUERING
correct = 0
print("\n========== DEL 8: EVALUERING ==========")
print("De første 10 prediksjonene:")

for i in range(len(X)):
    _, _, probability = forward_pass(X[i])
    predicted_class = 1 if probability >= 0.5 else 0
    if predicted_class == y[i]:
        correct += 1

    if i < 10:
        print(f"Protein {i + 1:2d}: Predikert = {predicted_class}, Fasit = {y[i]}")

accuracy = correct / len(X)
print(f"\nAccuracy på datasettet: {accuracy * 100:.2f}%")