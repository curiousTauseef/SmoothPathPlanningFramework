import configparser
import dill
import matplotlib.patches as mpatches
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from map import Map
from mvbspline import MVBSpline
from utils import load_map, load_spline

config = configparser.ConfigParser()
config.read('config.ini')

map_file = config.get('MAP', 'file')
map_resolution = config.getfloat('MAP', 'resolution')
pickle_file = config.get('PLOT', 'pickle_file')

# Initialize map
map_data = load_map(map_file)
map = Map(map_data, resolution=map_resolution)

# Open the pickle file and load the spline
bspline = load_spline(pickle_file)

# Get the properties of the spline
positions = np.array(list(bspline.calc_position(s) for s in bspline.s))
yaws = np.array(list(bspline.calc_yaw(s) * 180 / np.pi for s in bspline.s))
# Ensure that the yaw angles don't wrap around after hitting +/- pi
yaw_diffs = np.diff(yaws)
idxs = np.argwhere(abs(yaw_diffs) > 180)
for idx in idxs:
    sign = -1 if yaw_diffs[idx] > 0 else 1
    yaws[idx + 1] += sign * 360

curvatures = np.array(list(bspline.calc_curvature(s) for s in bspline.s))
derivative_curvatures = np.array(list(bspline.calc_curvature_derivative(s) for s in bspline.s))

# Plot
plt.figure()

# Plot position and boundaries
ax_pos = plt.subplot(2, 2, 1)
image =\
    plt.imshow(bspline._map.get_data(),
               extent=[0, bspline._map.get_width(), bspline._map.get_height(), 0])
pos_plt, = ax_pos.plot(positions[:, 0], positions[:, 1], '-r', linewidth=2, label='Cubic MVB')
start_marker, = ax_pos.plot(positions[0, 0], positions[0, 1], 'x',
                            color='lime', markersize=7, markeredgewidth=3, label="Start")
end_marker, = ax_pos.plot(positions[-1, 0], positions[-1, 1], 'o',
                          color='lime', markersize=7, markeredgewidth=3, label="End")

plt.xlabel('x (m)', fontsize=10)
plt.ylabel('y (m)', fontsize=10)
cmap = pl.get_cmap(image.cmap)
patches = [mpatches.Patch(color=cmap(Map.AVAILABLE), label='Available'),
           mpatches.Patch(color=cmap(Map.FREE), label='Free'),
           mpatches.Patch(color=cmap(Map.OCCUPIED), label='Occupied'),
           pos_plt,
           start_marker,
           end_marker]
# pl.plot(0, 0, "-", c=cmap(Map.AVAILABLE), label="Available")
# pl.plot(0, 0, "-", c=cmap(Map.OCCUPIED), label="Occupied")
# pl.plot(0, 0, "-", c=cmap(Map.FREE), label="Free")
plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))

# Plot yaw
ax_yaw = plt.subplot(2, 2, 2)
plt.plot(bspline.s, yaws, '-r', linewidth=2)
plt.xlabel('Arc Length (m)', fontsize=10)
plt.ylabel('Yaw (deg)', fontsize=10)
yaw_min, yaw_max = ax_yaw.get_ylim()
tick_period = 45
yaw_min = round(yaw_min / tick_period) * tick_period
yaw_max = round(yaw_max / tick_period) * tick_period
ax_yaw.set_yticks(np.arange(yaw_min, yaw_max + tick_period, tick_period))

# Plot curvature
plt.subplot(2, 2, 3)
plt.plot(bspline.s, curvatures, '-r', linewidth=2)
plt.xlabel('Arc Length (m)', fontsize=10)
plt.ylabel('Curvature (1/m)', fontsize=10)

# Plot derivative of curvature
plt.subplot(2, 2, 4)
plt.plot(bspline.s, derivative_curvatures, '-r', linewidth=2)
plt.xlabel('Arc Length (m)', fontsize=10)
plt.ylabel(r'Curvature Derivative (1/$m^2$)', fontsize=10)
knot_pts = bspline.get_knot_vector()
s_knots = list(bspline.get_knot_arc_length(t) for t in knot_pts)
# Get the curvature derivative at these knot points
knots_curv_deriv = np.array(list(bspline.calc_curvature_derivative(s) for s in s_knots))
plt.plot(s_knots, knots_curv_deriv, 'b.', linewidth=2, markersize=7, label='knots')

plt.legend()
plt.show()
