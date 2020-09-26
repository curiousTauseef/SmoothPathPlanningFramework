import bisect as bisector
import bspline
import bspline.splinelab as splinelab
import math
import numpy as np
import pickle
from scipy.interpolate import *
from scipy.integrate import *
from scipy.optimize import *


class MVBSpline:
    """
    BSpline for C0, C1, C2 and C3 continuity with minimum curvature variation
    """

    def __init__(self, control_points_ref, degree, map):
        self._map = map

        self._degree = degree # Degree of B-Spline

        # The control points should be a numpy array of size 2 x Num Control Points
        self._control_pts = control_points_ref
        self._num_control_pts = self._control_pts[1].size
        self._optimization_result = None

        self._num_knots = self._num_control_pts + self._degree + 1
        self._knot_max = 100
        self._knot_vec = np.linspace(0, self._knot_max, self._num_knots - 2 * self._degree)
        self._knot_vec = splinelab.augknt(self._knot_vec, self._degree)

        self._bspline = bspline.Bspline(self._knot_vec, self._degree)

        # Pre-calculate lambda functions for the first, second and third derivatives of the B-Spline bases
        self._n_d = self._bspline.diff(order=1)
        self._n_dd = self._bspline.diff(order=2)
        self._n_ddd = self._bspline.diff(order=3)

        # Define functions for calculating the x-y coordinates of the spline and their derivatives
        # Note that control points are 2-dimensional, so the output of these functions
        # is a 2-D vector where the first element represents the x dimension and the second is the y-dimension
        self._b = lambda ctrl_pts, t: np.sum(ctrl_pts * self._bspline(t), axis=1)
        self._b_d = lambda ctrl_pts, t: np.sum(ctrl_pts * self._n_d(t), axis=1)
        self._b_dd = lambda ctrl_pts, t: np.sum(ctrl_pts * self._n_dd(t), axis=1)
        self._b_ddd = lambda ctrl_pts, t: np.sum(ctrl_pts * self._n_ddd(t), axis=1)

        # Initialize Boundary Conditions
        self._s = None
        self._e = None
        self._d_s = None
        self._d_e = None

        # Store result of optimization
        self._success = False

        self._parameterize_arc_length()

    def _parameterize_arc_length(self):
        # Calculate the cumulative arc length at
        # points along the curve
        min, max = self.get_knot_limits()
        self.s = [0]
        self.t = np.linspace(min, max, 10000)
        # for knot in np.unique(self._knot_vec):
        #     if knot != min and knot != max and knot < max:
        #         self.t = np.insert(self.t, bisector.bisect(self.t, knot), knot)
        last_pos = None
        for i in range(len(self.t)):
            pos = self._b(self._control_pts, self.t[i])
            if last_pos is not None:
                ds = np.linalg.norm(pos - last_pos)
                self.s.append(self.s[-1] + ds)
            last_pos = pos

    def _get_index(self, s):
        idx = bisector.bisect(self.s, s)
        if idx == 0:
            return idx
        else:
            return idx - 1

    def get_knot_arc_length(self, t):
        idx = bisector.bisect(self.t, t)
        if idx == 0:
            idx = 0
        else:
            idx -= 1
        return self.s[idx]

    def calc_position(self, s):
        idx = self._get_index(s)
        pos = self._b(self._control_pts, self.t[idx])
        return pos

    def calc_curvature(self, s):
        idx = self._get_index(s)
        d = self._b_d(self._control_pts, self.t[idx])
        dd = self._b_dd(self._control_pts, self.t[idx])

        curv = (dd[1]*d[0] - dd[0]*d[1]) / ((d[0]**2 + d[1]**2)**1.5)

        return curv

    def calc_curvature_derivative(self, s):
        idx = self._get_index(s)
        d = self._b_d(self._control_pts, self.t[idx])
        dd = self._b_dd(self._control_pts, self.t[idx])
        ddd = self._b_ddd(self._control_pts, self.t[idx])

        dt_dx = 1 / d[0]
        ddt_ddx = - (dd[0] / (d[0] ** 3))
        dddt_dddx = (3*d[0]*(dd[0] ** 2) - (d[0] ** 2)*ddd[0]) / (d[0] ** 6)

        # Calculate the arc-length derivative
        arc = np.sqrt((1 + (d[1] / d[0]) ** 2))

        # Calculate the derivative of the curvature
        curv_d = ((arc ** 3)*(dddt_dddx*d[1] + 3*ddt_ddx*dt_dx*dd[1] + ddd[1]*(dt_dx**3)) -
                  3*((ddt_ddx*d[1] + dd[1]*(dt_dx**2)) ** 2) * arc * (d[1] / d[0])) / \
                 (arc ** 6)
        return curv_d / arc

    def calc_yaw(self, s):
        idx = self._get_index(s)
        d = self._b_d(self._control_pts, self.t[idx])
        return np.arctan2(d[1], d[0])

    def get_position(self, t):
        return self._b(self._control_pts, t)

    def add_boundary_conditions(self, s=None, e=None,
                                d_s=None, d_e=None,
                                b_left=None, b_right=None):
        self._s = s
        self._e = e
        self._d_s = d_s
        self._d_e = d_e

    def is_valid(self):
        return self._success

    def solve(self):
        self._success = self._optimize_control_points()
        return self._success

    def get_knot_limits(self):
        return self._knot_vec[0], self._knot_vec[-1] - 1e-5

    def get_knot_vector(self):
        return self._knot_vec

    def set_control_points(self, ctrl_pts):
        """
        Condition for setting control points is that the same number of control points must be set
        as what was originally used for the constructor
        """
        if ctrl_pts.shape != self._control_pts.shape:
            raise("[set_control_points ]Invalid Operation! Shape and Size of Control Points matrix does not match"
                  "that pf original control points set")
        else:
            self._control_pts = ctrl_pts

    def get_control_points(self):
        return self._control_pts

    def _derivative_curvature(self, ctrl_pts, t):
        """
        Calculates arc-length derivative of curvature of the b-spline
        at a given point on the curve
        """
        # Reshape ctrl pts to a 2d array
        ctrl_pts = np.reshape(ctrl_pts, (2, self._num_control_pts))

        # Calculate the first, second and third derivatives at the given spline position
        d = self._b_d(ctrl_pts, t)
        dd = self._b_dd(ctrl_pts, t)
        ddd = self._b_ddd(ctrl_pts, t)

        # Calculate derivatives of t w.r.t x
        dt_dx = 1 / d[0]
        ddt_ddx = - (dd[0] / (d[0] ** 3))
        dddt_dddx = (3*d[0]*(dd[0] ** 2) - (d[0] ** 2)*ddd[0]) / (d[0] ** 6)

        # Calculate the arc-length derivative
        arc = np.sqrt((1 + (d[1] / d[0]) ** 2))

        # Calculate the numerator of the arc-length derivative of curvature term that defines the cost function
        curv_d = ((arc ** 3)*(dddt_dddx*d[1] + 3*ddt_ddx*dt_dx*dd[1] + ddd[1]*(dt_dx**3)) -
                  3*((ddt_ddx*d[1] + dd[1]*(dt_dx**2)) ** 2) * arc * (d[1] / d[0])) / \
                 (arc ** 6)

        # Calculate the arc-length derivative of curvature
        arc_d_curve = abs(d[0]) * (curv_d ** 2) / arc
        # print(arc_d_curve)
        return arc_d_curve

    def _calculate_cost(self, ctrl_pts, t_start, t_end):
        """
        Calculates cost associated with the arc-length derivative of curvature
        through a numerical integration scheme
        """
        # Integrate the squared values of the arc-length derivative of the curvature
        # over the entire length of the 2-D B-Spline curve
        curvature_cost = romberg(lambda t: self._derivative_curvature(ctrl_pts, t),
                                 t_start,
                                 t_end)
        # print(np.reshape(ctrl_pts, (2, self._num_control_pts)))
        # print("---------------------------------------------------------")
        # print(curvature_cost)
        return curvature_cost

    def _calc_dist_2(self, ctrl_pts, point, t):
        """
        Calculate distance squared value between point on curve at parameter t and
        a given point
        """
        # Reshape ctrl points
        if not ctrl_pts[0].size == 2:
            ctrl_pts = np.reshape(np.array(ctrl_pts), (2, self._num_control_pts))
        return np.sum(np.power(self._b(ctrl_pts, t) - point, 2))

    def _calc_tangent(self, ctrl_pts, t):
        """
        Calculate tangent of B-Spline curve based on given control points at the parameter value t
        """
        # Reshape ctrl points
        if not ctrl_pts[0].size == 2:
            ctrl_pts = np.reshape(np.array(ctrl_pts), (2, self._num_control_pts))
        grad = self._b_d(ctrl_pts, t)
        return grad[1] / grad[0]

    def _check_bounds(self, ctrl_pts, t_start, t_end):
        if not ctrl_pts[0].size == 2:
            ctrl_pts = np.reshape(np.array(ctrl_pts), (2, self._num_control_pts))

        # Get representative points along the B-Spline with these control points and
        # check if they respect the given boundary
        pts = np.array(list(self._b(ctrl_pts, t) for t in np.linspace(t_start, t_end, 100)))

        nearest_non_free_dist = np.inf
        for pt in pts:
            if not self._map.is_available(pt, check_bounds=True):
                return -0.01
            #     cells, dist = self._map.get_available_cells_in_radius(pt, self._map.get_width())
            #     if len(dist) != 0:
            #         return min(-0.01, -min(dist))
            #     else:
            #         return -np.inf
            # else:
            #     cells, dist = self._map.get_available_cells_in_radius(pt, self._map.get_width(), invert=True)
            #     if len(dist) != 0:
            #         dist = min(dist)
            #         if dist < nearest_non_free_dist:
            #             nearest_non_free_dist = dist

        return 0.01

    def _optimize_control_points(self):
        """
        Runs optimization routine(s) to solve for the control points of the B-Spline that
        meet the given constraints and minimize the curvature variation
        """
        # There is a bug in the BSpline library where the Basis values are all 0 at the end point of the knots, where it
        # should actually have the last element to be 1. So, we don't go all the way to the end point, but get extremely
        # close to it
        max_t = self._knot_max - 1e-5

        # Add constraints
        constraints = []
        if self._s is not None:
            print("Start Point: {0}".format(self._s))
            constraints.append({'type': 'eq', 'fun': lambda ctrl_pts: self._calc_dist_2(ctrl_pts, self._s, 0)})
        if self._e is not None:
            print("End Point: {0}".format(self._e))
            constraints.append({'type': 'eq', 'fun': lambda ctrl_pts: self._calc_dist_2(ctrl_pts, self._e, max_t)})
        if self._d_s is not None:
            print("Gradient Start: {0}".format(self._d_s))
            constraints.append({'type': 'eq', 'fun': lambda ctrl_pts: self._calc_tangent(ctrl_pts, 0) - self._d_s})
        if self._d_e is not None:
            print("Gradient End: {0}".format(self._d_e))
            constraints.append({'type': 'eq', 'fun': lambda ctrl_pts: self._calc_tangent(ctrl_pts, max_t) - self._d_e})

        constraints.append({'type': 'ineq', 'fun': lambda ctrl_pts:
                            self._check_bounds(ctrl_pts, 0, max_t)})

        constraints = tuple(constraints)

        optimization_result = minimize(lambda ctrl_pts: self._calculate_cost(ctrl_pts, 0, max_t),
                                       self._control_pts, method='SLSQP',
                                       constraints=constraints, options={'maxiter': 150,
                                                                         'disp': True})

        # Use the returned control points to get the x and y coordinates of the B-Spline
        self._control_pts = np.reshape(np.array(optimization_result.x), (2, self._num_control_pts))

        # Update the arc length-parameteric representation of the B-Spline
        self._parameterize_arc_length()

        return optimization_result.success
