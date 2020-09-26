"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame]
(https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame]
(https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""
import bisect
import configparser
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import sys
import os

#######################################
ROOT_DIR = os.path.abspath("")
sys.path.append(ROOT_DIR)
print("*** ROOT_DIR: {} ***".format(ROOT_DIR))
#######################################

from mvbspline import MVBSpline
from utils import load_spline

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../QuinticPolynomialsPlanner/")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../CubicSpline/")

try:
    from quintic_polynomials_planner import QuinticPolynomial
    import cubic_spline_planner
except ImportError:
    raise

# Load Configuration Settings
config = configparser.ConfigParser()
config.read('config.ini')

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 3.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 10.0  # maximum curvature [1/m]
MAX_ROAD_WIDTH = 7.0  # maximum road width [m]
D_ROAD_W = 1.0  # road width sampling length [m]
DT = 0.2  # time tick [s]
MAX_T = 5  # max prediction time [m]
MIN_T = 4  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
K_J = 0.1
K_T = 0.1
K_D = 1.0
K_LAT = 1.0
K_LON = 1.0

show_animation = True


class QuarticPolynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, time):
        # calc coefficient of quartic polynomial

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * time ** 2, 4 * time ** 3],
                      [6 * time, 12 * time ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * time,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def get_t(self, s):
        t_vals = np.roots([self.a4, self.a3, self.a2, self.a1, self.a0 - s])
        if len(t_vals[np.real(t_vals) > 0]) == 0:
            return None
        else:
            return min(t_vals[np.real(t_vals) > 0])

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t ** 2 + \
             self.a3 * t ** 3 + self.a4 * t ** 4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
             3 * self.a3 * t ** 2 + 4 * self.a4 * t ** 3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t ** 2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class FrenetPath:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.cf = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):
    frenet_paths = []
    # knots = [0,
    #          6.150178141739622, 12.14687250919194, 18.029345216001566, 22.609826030952444,
    #          26.18506034880836, 29.02532472813526, 31.547540064865053, 39.97268344493031,
    #          52.315405322289386, 71.04022258278412, 83.71905294031018, 91.14858688633495,
    #          100.38886691076127, 110.09914934794392, 118.62906981288758, 127.308933951322,
    #          135.94710842375213] # Cubic
    # knots = [14.767923675545484, 19.774461326305445, 24.37097437943865,
    #          28.57249517731419, 31.984926666613504, 35.821445388706266, 43.88941911450072,
    #          59.64886726809787, 76.54096301731266, 85.74736093778719, 93.80968225090925,
    #          103.58597649088018, 113.83032270610506, 123.08243394415734, 133.8648008821973] # Quartic
    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MIN_T, MAX_T, DT):
            fp = FrenetPath()

            # lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)
            lat_qp = QuinticPolynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Longitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE,
                                TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = QuarticPolynomial(s0, c_speed, 0.0, tv, 0.0, Ti)
                # t_knots = np.array(list(lon_qp.get_t(knot_s) for knot_s in knots))
                # t_knots = t_knots[np.isreal(t_knots)]
                # t_knot = None
                # if len(t_knots) != 0:
                #     t_knot = np.real(min(t_knots))
                #
                # if t_knot and Ti >= t_knot >= 0.0:
                #     # Find proper index to put this time in the fp.t vector
                #     idx = bisect.bisect(fp.t, t_knot)
                #     tfp.t = list(np.insert(np.array(fp.t), idx, t_knot))
                #     tfp.d = list(np.insert(np.array(fp.d), idx, lat_qp.calc_point(t_knot)))
                #     tfp.d_d = list(np.insert(np.array(fp.d_d), idx, lat_qp.calc_first_derivative(t_knot)))
                #     tfp.d_dd = list(np.insert(np.array(fp.d_dd), idx, lat_qp.calc_second_derivative(t_knot)))
                #     tfp.d_ddd = list(np.insert(np.array(fp.d_ddd), idx, lat_qp.calc_third_derivative(t_knot)))
                tfp.s = [lon_qp.calc_point(t) for t in tfp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in tfp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in tfp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in tfp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1]) ** 2

                tfp.cd = K_J * Jp + K_T * Ti + K_D * tfp.d[-1] ** 2
                tfp.cv = K_J * Js + K_T * Ti + K_D * ds
                tfp.cf = K_LAT * tfp.cd + K_LON * tfp.cv

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):
    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            i_yaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(i_yaw + math.pi / 2.0)
            fy = iy + di * math.sin(i_yaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.hypot(dx, dy))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            if fp.ds[i] > 0:
                fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])
            else:
                fp.c.append(0)

    return fplist


def check_collision(fp, ob):
    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0]) ** 2 + (iy - ob[i, 1]) ** 2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS ** 2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):
    ok_ind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            print("max speed")
            continue
        elif any([abs(a) > MAX_ACCEL for a in
                  fplist[i].s_dd]):  # Max accel check
            print("max accel")
            continue
        elif any([abs(c) > MAX_CURVATURE for c in
                  fplist[i].c]):  # Max curvature check
            print("max curv:{}".format(fplist[i].c))
            print("i:{}".format(i))
            print("------------------")
            continue
        elif not check_collision(fplist[i], ob):
            print("collision")
            continue

        ok_ind.append(i)

    return [fplist[i] for i in ok_ind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):
    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    min_cost = float("inf")
    best_path = None
    for fp in fplist:
        if min_cost >= fp.cf:
            min_cost = fp.cf
            best_path = fp

    return best_path


def generate_target_course(x, y):
    csp = load_spline(config.get('FRENET', 'pickle_file'))
    csp._parameterize_arc_length()
    # csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp


def main():
    print(__file__ + " start!!")

    title = 'Quartic MVBSpline'

    # way points
    # wx = [0.0, 10.0, 20.5, 35.0, 70.5]
    # wy = [0.0, -6.0, 5.0, 6.5, 0.0]
    wx = [20, 67, 67, 45, 45, 61, 61, 50, 43]
    wy = [90, 90, 60, 60, 40, 40, 19, 20, 20]

    # obstacle lists
    # mvb cubic
    # ob = np.array([[39.36, 88.3],
    #                [61, 81.68],
    #                [57.44, 61.13],
    #                [48.95, 43.60],
    #                [58.8, 23.6]
    #                ])
    #mvb quartic
    ob = np.array([[39.36, 88.3],
                   [61, 79.2],
                   [57.44, 61.13],
                   [48.95, 43.60],
                   [57.72, 24.69]
                   ])
    # og cubic
    # ob = np.array([[31.14, 93.2],
    #                [67.86, 87.4],
    #                [64.295, 59.2],
    #                [59.36, 18.39],
    #                ])

    tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

    # initial state
    c_speed = 10.0 / 3.6  # current speed [m/s]
    c_d = 2.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current lateral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 20.0  # animation area length [m]

    x = []
    y = []
    s = []
    d = []
    curv = []
    longacc, longjerk = [], []
    latacc, latjerk = [], []
    for i in range(SIM_LOOP):
        path = frenet_optimal_planning(
            csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)
        # Record the current position of the robot
        # in the trajectory
        s0 = path.s[1]
        c_d = path.d[1]
        c_d_d = path.d_d[1]
        c_d_dd = path.d_dd[1]
        c_speed = path.s_d[1]

        s.append(path.s[1])
        x.append(path.x[1])
        y.append(path.y[1])
        d.append(path.d[1])
        longacc.append(path.s_dd[1]), longjerk.append(path.s_ddd[1])
        latacc.append(path.d_dd[1]), latjerk.append(path.d_ddd[1])
        curv.append(path.c[0])
        print("s: {0}, x: {1}, y: {2}".format(path.s[0], path.x[0], path.y[0]))
        if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 2.0:
            print("Goal")
            break

        if show_animation:  # pragma: no cover
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect(
                'key_release_event',
                lambda event: [exit(0) if event.key == 'escape' else None])
            # map
            plt.imshow((plt.imread(config.get('MAP', 'png'))), origin='lower')
            # obstacles
            plt.plot(ob[:, 0], ob[:, 1], "xk", markersize=5)
            # reference path
            plt.plot(tx, ty, '-r', label=title, marker='o', linestyle='dashed', linewidth=0.75,
                     markersize=2, alpha=0.25)
            # frenet path
            plt.plot(path.x[1:], path.y[1:], "-ob", label="Frenet", marker='o', linestyle='dashed',
                     linewidth=0.75, markersize=3)
            # robot
            plt.plot(path.x[0], path.y[0], "vb", label="Robot", markersize=10)
            # plt.xlim(path.x[1] - area, path.x[1] + area)
            plt.ylim(100, 0)
            plt.title(title + " Simulation \n v[km/h]:" + str(c_speed * 3.6)[0:4])
            plt.legend()
            plt.grid(True)
            plt.pause(0.0001)

    print("Finish")

    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    d_d = np.gradient(d) / np.gradient(s)
    d_dd = np.gradient(d_d) / np.gradient(s)
    d_theta = np.arctan2(dy, dx) - np.array(list(csp.calc_yaw(i) for i in s))
    k_r = np.array(list(csp.calc_curvature(i) for i in s))
    k_r_d = np.array(list(csp.calc_curvature_derivative(i) for i in s))
    curvature = (d_dd + (k_r_d + k_r * d)*np.sin(d_theta)*(np.cos(d_theta) ** 2)) / ((1 - k_r*d) ** 2) + \
                (k_r * np.cos(d_theta)) / (1 - k_r * d)
    # curvature = (ddy * dx - dy * ddx) / ((dx ** 2 + dy ** 2) ** 1.5)
    # curvature = np.gradient(dy / dx) / ((1 + (dy/dx)**2) ** 1.5)
    curvature[0] = curvature[1] = curvature[2]
    curvature[-1] = curvature[-2] = curvature[-3]

    plt.figure()
    # map
    plt.imshow((plt.imread(config.get('MAP', 'png'))), origin='lower')
    # obstacles
    plt.plot(ob[:, 0], ob[:, 1], "xk", markersize=5)
    # reference path
    plt.plot(tx, ty, '-r', label=title, marker='o', linestyle='dashed', linewidth=0.75,
             markersize=2, alpha=0.1)
    # frenet path
    plt.plot(x, y, "-ob", label="Frenet", marker='o', linestyle='dashed',
             linewidth=0.75, markersize=3)
    # robot
    # plt.plot(path.x[0], path.y[0], "vb", label="Robot", markersize=10)
    # plt.xlim(path.x[1] - area, path.x[1] + area)
    plt.ylim(100, 0)
    plt.title(title)
    plt.legend()
    plt.grid(True)


    # Based on the actual trajectory of the robot show
    fig, ((ax1, ax2)) = plt.subplots(1, 2)
    fig.suptitle(title)

    pad = 0.15
    #curvature
    # ax1.plot(csp.s, list(csp.calc_curvature(s) for s in csp.s), '-r', label="Quartic B-Spline", marker='o', linestyle='dashed',
    #          linewidth=0.75, markersize=2, alpha=0.25)
    ax1.plot(s, list(csp.calc_curvature(i) for i in s), '-r', label=title, marker='o', linestyle='dashed',
             linewidth=.75, markersize=3)
    ax1.plot(s, curvature, '-b', label="Frenet", marker='o', linestyle='dashed',
                     linewidth=0.75, markersize=3)
    # ax1.set_ylim(-abs(max(curvature))-pad, abs(max(curvature))+pad)
    ax1.set_title(r"Curvature")
    ax1.set_ylabel(r"${\kappa}$ $[\dfrac{1}{m}]$")
    ax1.legend()

    # ax2.plot(csp.s, list(csp.calc_curvature_derivative(s) for s in csp.s), '-r', label="Quartic B-Spline", marker='o',
    #          linestyle='dashed', linewidth=0.75, markersize=2, alpha=0.25)
    ax2.plot(s, list(csp.calc_curvature_derivative(i) for i in s), '-r', label=title, marker='o',
             linestyle='dashed', linewidth=.75, markersize=3)
    ax2.plot(s, np.gradient(curvature)/np.gradient(s), '-b', label="Frenet", marker='o', linestyle='dashed',
                     linewidth=0.75, markersize=3)
    # ax2.set_ylim(-abs(max(curvature))-pad, abs(max(curvature))+pad)
    ax2.set_title(r"Derivative of Curvature")
    ax2.set_ylabel(r"$\dot{\kappa}$ $[\dfrac{1}{m^2}]$")
    ax2.legend()

    # ax3.plot(s, latacc, '-b', label="Acceleration", marker='o', linestyle='dashed',
    #                  linewidth=0.75, markersize=3)
    # ax3.set_ylabel(r"$a$ $[\dfrac{m}{s^2}]$", color='b')
    # ax3.set_xlabel(r"$Arc$ $Length$ $[m]$")
    # # ax3.set_ylim(-abs(max(latjerk)) - pad, abs(max(latjerk)) + pad)
    # ax3.set_ylim(-abs(MAX_ACCEL) - pad, abs(MAX_ACCEL) + pad)
    # ax4 = ax3.twinx()
    # ax4.plot(s, latjerk, '-m', label="Jerk", marker='o', linestyle='dashed',
    #                  linewidth=0.75, markersize=3)
    # # ax4.set_ylim(-abs(max(latjerk)) - pad, abs(max(latjerk)) + pad)
    # ax4.set_ylim(-abs(MAX_ACCEL) - pad, abs(MAX_ACCEL) + pad)
    # ax4.set_title("Lateral Frenet Trajectory")
    # # ask matplotlib for the plotted objects and their labels
    # lines, labels = ax3.get_legend_handles_labels()
    # lines2, labels2 = ax4.get_legend_handles_labels()
    # ax4.legend(lines + lines2, labels + labels2, loc=0)
    #
    # ax5.plot(s, longacc, '-b', label="Acceleration", marker='o', linestyle='dashed',
    #                  linewidth=0.75, markersize=3)
    # # ax5.set_ylim(-abs(max(latjerk)) - pad, abs(max(latjerk)) + pad)
    # ax5.set_ylim(-abs(MAX_ACCEL) - pad, abs(MAX_ACCEL) + pad)
    # ax5.set_xlabel(r"$Arc$ $Length$ $[m]$")
    # ax6 = ax5.twinx()
    # ax6.plot(s, longjerk, '-m', label="Jerk", marker='o', linestyle='dashed',
    #                  linewidth=0.75, markersize=3)
    # # ax6.set_ylim(-abs(max(latjerk)) - pad, abs(max(latjerk)) + pad)
    # ax6.set_ylim(-abs(MAX_ACCEL) - pad, abs(MAX_ACCEL) + pad)
    # ax6.set_ylabel(r"$\dot{a}$ $[\dfrac{m}{s^3}]$", color='m')
    # ax6.set_title("Longitudinal Frenet Trajectory")
    # # ask matplotlib for the plotted objects and their labels
    # lines, labels = ax5.get_legend_handles_labels()
    # lines2, labels2 = ax6.get_legend_handles_labels()
    # ax6.legend(lines + lines2, labels + labels2, loc=0)

    plt.ioff()
    plt.show()
    pass

if __name__ == '__main__':
    main()
