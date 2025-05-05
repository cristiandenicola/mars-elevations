import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def render_3d(array):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    x = np.arange(array.shape[1])
    y = np.arange(array.shape[0])
    x, y = np.meshgrid(x, y)

    ax.plot_surface(x, y, array, cmap='terrain', linewidth=0, antialiased=False)
    ax.set_title('3D Elevation Map')
    plt.tight_layout()
    plt.show()
