import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# DEL 1: Last ned MNIST og gjør bildene om til tall.
torch.manual_seed(42)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# DEL 2: MLP - gjør 28x28-bildet om til 784 tall.
mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)

# DEL 3: CNN - finner mønstre i bildene med filtre.
cnn = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Conv2d(8, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 10)
)

# DEL 4: Tren en modell og mål tiden.
def train_model(model, name):
    print('\n========== ' + name + ' ==========')
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()
    losses = []
    start = time.time()

    for epoch in range(3):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()  # PyTorch regner ut backpropagation.
            optimizer.step()
            total_loss += loss.item() * len(images)

        average_loss = total_loss / len(train_data)
        losses.append(average_loss)
        print('Epoch', epoch + 1, '- Loss:', round(average_loss, 4))

    training_time = time.time() - start
    return losses, training_time

# DEL 5: Test hvor mange sifre modellen gjetter riktig.
def test_model(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
    return 100 * correct / total

mlp_losses, mlp_time = train_model(mlp, 'DEL 2: MLP')
cnn_losses, cnn_time = train_model(cnn, 'DEL 3: CNN')
mlp_accuracy = test_model(mlp)
cnn_accuracy = test_model(cnn)

mlp_parameters = 0
for parameter in mlp.parameters():
    mlp_parameters += parameter.numel()

cnn_parameters = 0
for parameter in cnn.parameters():
    cnn_parameters += parameter.numel()

print('\n========== DEL 5: RESULTATER ==========')
print('Model  Accuracy   Parameters   Training time')
print(f'MLP    {mlp_accuracy:.2f}%     {mlp_parameters}       {mlp_time:.1f} sec')
print(f'CNN    {cnn_accuracy:.2f}%     {cnn_parameters}         {cnn_time:.1f} sec')

# DEL 6: Tegn og lagre grafen til rapporten.
plt.plot(range(1, 4), mlp_losses, marker='o', label='MLP')
plt.plot(range(1, 4), cnn_losses, marker='o', label='CNN')
plt.xlabel('Epoch')
plt.ylabel('Training loss')
plt.title('MNIST: MLP vs CNN')
plt.xticks([1, 2, 3])
plt.legend()
plt.tight_layout()
plt.savefig('mnist_training.png')
plt.show()
