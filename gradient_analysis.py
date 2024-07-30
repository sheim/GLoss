# gradient_analysis.py
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Function to compare gradients
def compare_gradients(grads1, grads2):
    magnitudes1 = [torch.norm(g1).item() for g1 in grads1]
    magnitudes2 = [torch.norm(g2).item() for g2 in grads2]
    dot_products = [
        torch.dot(g1.view(-1), g2.view(-1)).item() for g1, g2 in zip(grads1, grads2)
    ]
    cosine_similarities = [
        torch.nn.functional.cosine_similarity(g1.view(1, -1), g2.view(1, -1)).item()
        for g1, g2 in zip(grads1, grads2)
    ]

    return magnitudes1, magnitudes2, dot_products, cosine_similarities


# Load gradient data from file
gradient_data = torch.load("gradient_data.pt")

# Lists to store data for plotting
all_magnitudes1 = []
all_magnitudes2 = []
all_dot_products = []
all_cosine_similarities = []

# Analyze gradients
for i, data in enumerate(gradient_data):
    input_data = data["input"]
    grads_loss1 = data["grads_loss1"]
    grads_loss2 = data["grads_loss2"]

    magnitudes1, magnitudes2, dot_products, cosine_similarities = compare_gradients(
        grads_loss1, grads_loss2
    )

    all_magnitudes1.append(magnitudes1)
    all_magnitudes2.append(magnitudes2)
    all_dot_products.append(dot_products)
    all_cosine_similarities.append(cosine_similarities)

# Convert lists to numpy arrays for easier plotting
all_magnitudes1 = np.array(all_magnitudes1)
all_magnitudes2 = np.array(all_magnitudes2)
all_dot_products = np.array(all_dot_products)
all_cosine_similarities = np.array(all_cosine_similarities)

# Plotting
iterations = range(len(gradient_data))
layers = range(len(grads_loss1))

plt.figure(figsize=(16, 12))

# Magnitudes
plt.subplot(3, 1, 1)
for layer in layers:
    plt.plot(
        iterations, all_magnitudes1[:, layer], label=f"Layer {layer+1} Magnitude 1"
    )
    plt.plot(
        iterations,
        all_magnitudes2[:, layer],
        label=f"Layer {layer+1} Magnitude 2",
        linestyle="dashed",
    )
plt.xlabel("Iteration")
plt.ylabel("Magnitude")
plt.legend()
plt.title("Gradient Magnitudes Over Iterations")

# Dot Products
plt.subplot(3, 1, 2)
for layer in layers:
    plt.plot(
        iterations, all_dot_products[:, layer], label=f"Layer {layer+1} Dot Product"
    )
plt.xlabel("Iteration")
plt.ylabel("Dot Product")
plt.legend()
plt.title("Gradient Dot Products Over Iterations")

# Cosine Similarities
plt.subplot(3, 1, 3)
for layer in layers:
    plt.plot(
        iterations,
        all_cosine_similarities[:, layer],
        label=f"Layer {layer+1} Cosine Similarity",
    )
plt.xlabel("Iteration")
plt.ylabel("Cosine Similarity")
plt.legend()
plt.title("Gradient Cosine Similarities Over Iterations")

plt.tight_layout()
plt.show()

# Optional: Heatmap for a single iteration
iteration_to_visualize = 0  # Change this to visualize a different iteration
plt.figure(figsize=(12, 6))
heatmap_data = np.array(
    [
        all_dot_products[iteration_to_visualize],
        all_cosine_similarities[iteration_to_visualize],
    ]
)
sns.heatmap(
    heatmap_data,
    annot=True,
    cbar=True,
    xticklabels=[f"Layer {i+1}" for i in layers],
    yticklabels=["Dot Product", "Cosine Similarity"],
)
plt.title(
    f"Heatmap of Dot Products and Cosine Similarities (Iteration {iteration_to_visualize})"
)
plt.show()
