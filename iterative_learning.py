# iterative_learning.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Define sizes
input_size = 2
output_size = 2
hidden_size = 8
train_size = 10
test_size = 5


# Simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=4):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Tanh()
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)


# Generate synthetic data with random distribution
torch.manual_seed(42)
all_data = torch.randn(train_size + test_size, input_size)
all_target = torch.randn(train_size + test_size, output_size)

# Sort data
sorted_indices = torch.argsort(all_data[:, 0])  # Sort based on the first column
all_data = all_data[sorted_indices]
all_target = all_target[sorted_indices]

# Split data into training and test sets
train_data = all_data[:train_size]
train_target = all_target[:train_size]
test_data = all_data[train_size:]
test_target = all_target[train_size:]

# Shuffle training data
shuffle_indices = torch.randperm(train_size)
train_data = train_data[shuffle_indices]
train_target = train_target[shuffle_indices]

# Instantiate the network
model = SimpleNN(input_size, output_size, hidden_size=hidden_size)

# Loss functions
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# To store gradient data and losses
gradient_data = []
train_losses1 = []
train_losses2 = []
test_losses1 = []
test_losses2 = []

# Iterative learning
for epoch in range(20):
    epoch_train_loss1 = 0
    epoch_train_loss2 = 0
    epoch_test_loss1 = 0
    epoch_test_loss2 = 0

    for i in range(train_data.size(0)):
        input_data = train_data[i].unsqueeze(0)
        target_data = train_target[i].unsqueeze(0)

        # Forward pass
        output = model(input_data)

        # Compute losses
        loss1 = criterion1(output, target_data)
        loss2 = criterion2(output, target_data)
        epoch_train_loss1 += loss1.item()
        epoch_train_loss2 += loss2.item()

        # Compute gradients for loss1
        optimizer.zero_grad()
        loss1.backward(retain_graph=True)
        grads_loss1 = [param.grad.clone() for param in model.parameters()]

        # Compute gradients for loss2
        optimizer.zero_grad()
        loss2.backward()
        grads_loss2 = [param.grad.clone() for param in model.parameters()]

        # Accumulate gradients from both losses
        for param, grad_loss1, grad_loss2 in zip(
            model.parameters(), grads_loss1, grads_loss2
        ):
            param.grad = grad_loss1 + grad_loss2

        # Store gradient data
        gradient_data.append(
            {
                "input": input_data,
                "grads_loss1": grads_loss1,
                "grads_loss2": grads_loss2,
            }
        )

        # Perform optimization step
        optimizer.step()

    # Track training losses
    train_losses1.append(epoch_train_loss1 / train_data.size(0))
    train_losses2.append(epoch_train_loss2 / train_data.size(0))

    # Track test losses
    with torch.no_grad():
        for i in range(test_data.size(0)):
            input_data = test_data[i].unsqueeze(0)
            target_data = test_target[i].unsqueeze(0)

            output = model(input_data)
            loss1 = criterion1(output, target_data)
            loss2 = criterion2(output, target_data)
            epoch_test_loss1 += loss1.item()
            epoch_test_loss2 += loss2.item()

    test_losses1.append(epoch_test_loss1 / test_data.size(0))
    test_losses2.append(epoch_test_loss2 / test_data.size(0))

# Save gradient data to a file
torch.save(gradient_data, "gradient_data.pt")
print("Gradient data saved to gradient_data.pt")

# Plot training and test losses
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(train_losses1, label="Training Loss 1 (MSE)")
plt.plot(test_losses1, label="Test Loss 1 (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss 1 (MSE) Over Epochs")

plt.subplot(1, 2, 2)
plt.plot(train_losses2, label="Training Loss 2 (MSE)")
plt.plot(test_losses2, label="Test Loss 2 (MSE)")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Test Loss 2 (MSE) Over Epochs")

plt.tight_layout()
plt.show()
