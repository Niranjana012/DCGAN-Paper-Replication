import numpy as np
import matplotlib.pyplot as plt

# Load recovered loss data
loss_data = np.load("loss_data_recovered.npz")
d_losses = loss_data["d_losses"]
g_losses = loss_data["g_losses"]

# Define number of epochs (from your training config)
num_epochs = 200  

# Convert batch index to epoch index for X-axis scaling
epochs = np.linspace(1, num_epochs, len(d_losses))  

# Plot the graph
plt.figure(figsize=(10, 5))
plt.plot(epochs, d_losses, label="Discriminator Loss", color="red")
plt.plot(epochs, g_losses, label="Generator Loss", color="blue")

# Labels & Title
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Recovered Training Losses of Generator and Discriminator")
plt.legend()
plt.grid()

# Save & show
plt.savefig("recovered_training_loss_plot.png")
plt.show()
