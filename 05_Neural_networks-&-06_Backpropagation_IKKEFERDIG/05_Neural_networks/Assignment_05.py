# IKT466 - Assignment 05: Neural Network from scratch
# Enkel og lettlest implementasjon av forward pass.
# Vi bruker ikke PyTorch eller TensorFlow.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# DEL 1: LAST INN DATA
# CSV-filen ligger i samme mappe som dette programmet.
fil = Path(__file__).with_name("ecoli.csv")
data = pd.read_csv(fil)

# Vi skal bare bruke proteinene som har klassen cp eller im.
data = data[data["SITE"].isin(["cp", "im"])].copy()

# Første kolonne er et navn, og siste kolonne er klassen.
# Alle kolonnene imellom er numeriske egenskaper.
kolonner = data.columns[1:-1]
X = data[kolonner].to_numpy(dtype=float)

# Gjør klassene om til tall: cp = 0 og im = 1.
y = data["SITE"].map({"cp": 0, "im": 1}).to_numpy()

print("========== DEL 1: DATA ==========")
print("Antall proteiner:", len(X))
print("Antall egenskaper:", X.shape[1])
print("Klasser: cp = 0, im = 1")


# DEL 2: NORMALISER DATA
# Min-max-normalisering gjør at hver egenskap ligger mellom 0 og 1.
minste = X.min(axis=0)
storste = X.max(axis=0)
forskjell = storste - minste

# Hvis en kolonne bare har én verdi, unngår vi deling på null.
forskjell[forskjell == 0] = 1
X = (X - minste) / forskjell

print("\n========== DEL 2: NORMALISERING ==========")
print("Første normaliserte protein:", np.round(X[0], 3))
print("Fasit for proteinet:", y[0])


# DEL 3: AKTIVERINGSFUNKSJONER
# Sigmoid gir et tall mellom 0 og 1.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU gjør negative tall til 0, og beholder positive tall.
def relu(x):
    return np.maximum(0, x)


# DEL 4: ÉN NEURON
# En neuron ganger input med vekter, legger til bias og bruker aktivering.
def neuron(inputs, weights, bias, activation):
    total = np.dot(inputs, weights) + bias
    output = activation(total)
    return output


# DEL 5: HIDDEN LAYER
# Et hidden layer består av flere neuroner.
def hidden_layer(inputs, weights, biases, activation):
    outputs = []

    for i in range(len(biases)):
        # Hver kolonne i weights inneholder vektene til én neuron.
        result = neuron(inputs, weights[:, i], biases[i], activation)
        outputs.append(result)

    return np.array(outputs)


# DEL 6: OUTPUT LAYER
# Output layer bruker resultatene fra hidden layer til å gi en prediksjon.
def output_layer(inputs, weights, bias):
    return neuron(inputs, weights, bias, sigmoid)


# DEL 7: TEST MED FASTE VEKTER
# Faste vekter gjør at vi får samme resultat hver gang vi kjører koden.
# Dette er bare en test av forward pass, ikke en trent modell.
inputs = X[0]

# Vi lager 3 neuroner i hidden layer.
# Det er én vekt per input-egenskap og per hidden neuron.
hidden_weights = np.full((len(inputs), 3), 0.1)
hidden_biases = np.array([0.1, 0.2, 0.3])

# Output layer har 3 vekter, fordi hidden layer har 3 neuroner.
output_weights = np.array([0.2, 0.3, 0.4])
output_bias = 0.1

print("\n========== DEL 7: FORWARD PASS ==========")

# Først tester vi en enkelt neuron.
single_output = neuron(inputs, hidden_weights[:, 0], 0.1, sigmoid)
print("En neuron (Sigmoid):", round(float(single_output), 4))

# Vi tester nettverket med både Sigmoid og ReLU i hidden layer.
for activation, name in [(sigmoid, "Sigmoid"), (relu, "ReLU")]:
    hidden_outputs = hidden_layer(inputs, hidden_weights, hidden_biases, activation)
    prediction = output_layer(hidden_outputs, output_weights, output_bias)
    predicted_class = 1 if prediction >= 0.5 else 0

    print("\nAktivering:", name)
    print("Hidden layer:", np.round(hidden_outputs, 4))
    print("Output:", round(float(prediction), 4))
    print("Predikert klasse:", predicted_class)
    print("Faktisk klasse:", y[0])


# DEL 8: GRAFER
# Vi tegner Sigmoid og ReLU for verdier mellom -5 og 5.
values = np.linspace(-5, 5, 200)

plt.plot(values, sigmoid(values), label="Sigmoid")
plt.plot(values, relu(values), label="ReLU")
plt.title("Activation Functions: Sigmoid vs ReLU")
plt.xlabel("Input")
plt.ylabel("Output")
plt.legend()
plt.grid(True)
plt.savefig(Path(__file__).with_name("activation_functions.png"))
plt.show()
