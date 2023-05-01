import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Sample data
num_datasets = 8
num_points = 100
amplitude = 5

datasets = [amplitude * np.sin(np.linspace(0, 2 * np.pi, num_points)) + i * 2 * amplitude for i in range(num_datasets)]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each dataset along the Z-axis
for i, data in enumerate(datasets):
    x = np.arange(num_points)
    y = np.ones(num_points) * i
    z = data
    ax.plot(x, y, z)
    ax.axis('off')



# Show the plot
plt.show()