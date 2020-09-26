import configparser
import matplotlib.pyplot as plt
import numpy as np
import dill
import pickle

from map import Map
from mvbspline import MVBSpline
from utils import load_map

config = configparser.ConfigParser()
config.read('config.ini')

# class Test:
#     def __init__(self, data):
#         self._data = np.array(data)
#
#     def __getitem__(self, item):
#         return self._data[item]

if __name__ == "__main__":
    # Map for planning
    map_file = config.get('MAP', 'file')
    map_resolution = config.getfloat('MAP', 'resolution')

    # Degree of spline and pickle filename to store spline object after optimization process
    degree = config.getint('CURVE', 'degree')
    filename = config.get('CURVE', 'save_filename')

    # Initialize map
    map_data = load_map(map_file)
    map = Map(map_data, resolution=map_resolution)

    # Set the cells in the map which are traversable
    map[10:40, 86:97] = Map.AVAILABLE
    map[40:90, 78:97] = Map.AVAILABLE
    map[59:74, 56:78] = Map.AVAILABLE
    map[39:59, 56:64] = Map.AVAILABLE
    map[39:55, 36:56] = Map.AVAILABLE
    map[55:65, 36:44] = Map.AVAILABLE
    map[57:65, 10:36] = Map.AVAILABLE
    map[38:57, 16:24] = Map.AVAILABLE

    # map[10:40, 86:97] = Map.AVAILABLE
    # map[40:90, 78:97] = Map.AVAILABLE
    # map[59:74, 56:78] = Map.AVAILABLE
    # map[67:98, 36:64] = Map.AVAILABLE
    # map[77:85, 24:36] = Map.AVAILABLE
    # map[66:98, 2:24] = Map.AVAILABLE

    # print(map.is_available(np.array([6.59, 4.12])))

    # Initial guess at a path which goes through the allowed area
    path = np.array([[20, 90],
                     [67, 90],
                     [67, 60],
                     [45, 60],
                     [45, 40],
                     [61, 40],
                     [61, 19],
                     [50, 20],
                     [43, 20]]) * map_resolution

    # # Find path satisfying minimum radius of curvature of robot and C3 continuous using a Quartic Spline
    # # Points for the left boundary
    # left_boundary_pts = np.array([[10, 13], [38, 13], [57, 24], [77, 63], [80, 100]])
    #
    # # Points for the right boundary
    # right_boundary_pts = np.array([[10, 2], [70, 2], [73, 35], [85, 65], [85, 80], [98, 80], [98, 98]])
    #
    # # Points for the initial control points guess (these should lie within the boundaries). They can be
    # # chosen randomly, as long as they are within the right and left boundaries
    # x_ctrl_pts = np.array([20, 23, 28, 30, 35, 41, 46, 50, 55, 58, 62, 69, 74, 76, 78, 80, 81, 85, 88, 90]) * map_resolution
    # y_ctrl_pts = map.get_height() - np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 35, 45, 50, 60, 70, 82, 85, 87, 90]) * map_resolution
    # ctrl_pts_guess = np.vstack((np.array(x_ctrl_pts), np.array(y_ctrl_pts)))
    # path = ctrl_pts_guess.transpose()
    # Show the map
    plt.imshow(map.get_data(), extent=[0, map.get_width(), map.get_height(), 0])
    plt.plot(path[:,0], path[:,1], '-.xr', linewidth=2)
    plt.show()
    quartic = MVBSpline(path.transpose(), degree, map)  # Set to 3 for cubic, 4 for quartic
    quartic.add_boundary_conditions(np.array([20, 90]), np.array([43, 20]), 0, 0)

    # Run this to calculate new control points for B-Spline (runs the optimization process which takes a while)
    success = quartic.solve()

    # Save the spline object to be reused later
    quartic.__module__ = 'mvbspline'
    fh = open(filename, 'w')
    pickle.dump(quartic, fh)
    fh.close()

    # Some examples to plot different properties of the B-Spline

    # Boundaries and Curve
    knot_min, knot_max = quartic.get_knot_limits()
    points = np.array(list(quartic.get_position(t) for t in np.linspace(knot_min, knot_max, 1000)))
    plt.figure()
    plt.imshow(map.get_data(), extent=[0, map.get_width(), map.get_height(), 0])
    plt.plot(points[:, 0], points[:, 1])

    # Curvature vs Arc Length
    # plt.figure()
    # curvatures = np.array(list(quartic.calc_curvature(s) for s in quartic.s))
    # plt.plot(quartic.s, curvatures)

    # Derivative of Curvature vs Arc Length
    plt.figure()
    curvatures_d = np.array(list(quartic.calc_curvature_derivative(s) for s in quartic.s))
    plt.plot(quartic.s, curvatures_d)

    # Yaw vs Arc Length
    # plt.figure()
    # yaws = np.array(list(quartic.calc_yaw(s) for s in quartic.s))
    # plt.plot(quartic.s, yaws * 180/np.pi)

    # Show all plots
    plt.show()
