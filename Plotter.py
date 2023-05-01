import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import math

from matplotlib import gridspec
class RipplePlot:

    def __init__(self, axes_handle):
        self.ax = axes_handle
        self.num_datasets = 8
        self.num_points = 100
        self.amplitude = 5

        datasets = [self.amplitude * np.sin(np.linspace(0, 2 * np.pi, self.num_points)) + i * 2 * self.amplitude for i in range(self.num_datasets)]

        # Plot each dataset along the Z-axis
        for i, data in enumerate(datasets):
            x = np.arange(self.num_points)
            y = np.ones(self.num_points) * i
            z = data
            self.ax.plot(x, y, z)
            self.ax.axis('off')

N = 8
cols = 4
rows = int(math.ceil(N / cols))

gs = gridspec.GridSpec(rows, cols)
fig = plt.figure()
for n in range(N):
    ax = fig.add_subplot(gs[n], projection='3d')
    RipplePlot(ax)

fig.tight_layout()
plt.show()