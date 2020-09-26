import numpy as np

class Map:
    FREE = 255
    AVAILABLE = 128
    OCCUPIED = 0

    def __init__(self, data, resolution=0.1, d_theta=np.pi/8):
        self._data = np.array(data).transpose()
        self._resolution = resolution
        self._d_theta = d_theta
        self._width = self._data.shape[0]   # x
        self._height = self._data.shape[1]  # y

    def __setitem__(self, item, value):
        self._data[item] = value

    def __getitem__(self, item):
        return self._data[item]

    def get_width(self):
        return self._width * self._resolution

    def get_height(self):
        return self._height * self._resolution

    def get_data(self):
        return self._data.transpose().copy()

    def get_inflated_obstacles_data(self, inflation_radius=1.0):
        data = self._data.copy()
        data_ref = self._data.copy()

        inflate_cells = np.ceil(inflation_radius / self._resolution)
        # if np.mod(inflate_cells, 2) == 0:
        #     inflate_cells += 1
        # Kernel dimensions should be odd-numbered values
        kernel = (int(2 * inflate_cells + 1), int(2 * inflate_cells + 1))

        # Apply one-cell width, zero padding at the boundaries of the array
        x_padding = int(inflate_cells)
        y_padding = int(inflate_cells)
        data_ref = np.pad(self._data, ((x_padding, x_padding), (y_padding, y_padding)))

        # To inflate the obstacles, we have to
        # apply a dilation operation by running a
        # max filter with a kernel in a convolution operation
        for x in range(x_padding, data_ref.shape[0] - x_padding):
            for y in range(y_padding, data_ref.shape[1] - y_padding):
                data[x - x_padding, y - y_padding] = \
                    np.min(data_ref[x - x_padding: x + x_padding + 1,
                                    y - y_padding: y + y_padding + 1])

        return data.transpose()

    def get_resolution(self):
        return self._resolution

    def get_delta_theta(self):
        return self._d_theta

    def get_obstacle_boundaries(self):
        data = np.vstack((np.zeros((1, self._width)), self._data))
        data = np.hstack((np.zeros((self._height + 1, 1)), data))
        print(data.shape)
        grad_x = np.diff(data, axis=1)
        grad_y = np.diff(data, axis=0)

        boundary_cells = np.empty((0, 2), dtype=int)

        # Indexes of occupied to free boundary cells
        for axis in (0, 1):
            grad = np.diff(self._data, axis=axis)
            obs_to_free_idxs = np.argwhere(grad == Map.OCCUPIED - Map.FREE)
            free_to_obs_idxs = np.argwhere(grad == Map.FREE - Map.OCCUPIED)
            # For the free to obstacle transitions, increment index by 1
            obs_to_free_idxs[:, axis] += 1
            boundary_cells = np.vstack((boundary_cells, obs_to_free_idxs, free_to_obs_idxs))

        return np.unique(boundary_cells, axis=0)

    def worldToMap(self, pos):
        return np.floor(pos / self._resolution).astype(int)

    def mapToWorld(self, pos):
        return (self._resolution / 2) + self._resolution * pos

    def is_occupied(self, pos, world_coords=True, check_bounds=False):
        if world_coords:
            pos = self.worldToMap(pos).astype(int)
        else:
            pos = pos.astype(int)

        if not check_bounds or self.in_bounds(pos, False):
            return self._data[pos[0], pos[1]] == Map.OCCUPIED
        else:
            return True

    def is_available(self, pos, world_coords=True, check_bounds=False):
        if world_coords:
            pos = self.worldToMap(pos).astype(int)
        else:
            pos = pos.astype(int)

        if not check_bounds or self.in_bounds(pos, False):
            return self._data[pos[0], pos[1]] == Map.AVAILABLE
        else:
            return False

    def in_bounds(self, pos, world_coords=True):
        if world_coords:
            pos = self.worldToMap(pos).astype(int)
        else:
            pos = pos.astype(int)

        return self._width > pos[0] >= 0 and self._height > pos[1] >= 0

    def enforce_bounds(self, pos):
        """

        :param pos: Position in map coordinates
        :return:
        """
        if pos[0] < 0:
            pos[0] = 0
        elif pos[0] >= self._width:
            pos[0] = self._width - 1

        if pos[1] < 0:
            pos[1] = 0
        elif pos[1] >= self._height:
            pos[1] = self._height - 1

        return pos

    def find_available(self, pos, end, invert=False):
        # Rasterize the line using Bresenham's algorithm
        # Use DDA to cover all cells that the ray passes through (modified Bresenham)
        cur_cell = self.worldToMap(pos)
        pos = self.mapToWorld(cur_cell)
        end_cell = self.worldToMap(end)

        # Cells that the ray traverses through
        cells = np.empty((0, 2), dtype=int)

        x, y = cur_cell[0], cur_cell[1]
        x2, y2 = end_cell[0], end_cell[1]
        dx = abs(x2 - x)
        dy = abs(y2 - y)
        step_x = 1 if x2 - x > 0 else -1
        step_y = 1 if y2 - y > 0 else -1
        steep = True if dy > dx else False

        if steep:
            x, y = y, x
            dx, dy = dy, dx
            step_x, step_y = step_y, step_x

        prev_error = -dx
        error = (2 * dy) - dx
        for i in range(0, dx):
            cells_to_add = []
            x += step_x

            if steep:
                x_dist = self._height - 1 - x if step_x > 0 else x
                y_dist = self._width - 1 - y if step_y > 0 else y
            else:
                x_dist = self._width - 1 - x if step_x > 0 else x
                y_dist = self._height - 1 - y if step_y > 0 else y

            if error >= 0:
                y = y + step_y
                y_dist -= 1
                # Check if we went through the lower cell along y while climbing
                if prev_error + error < 0 and x_dist >= 0:
                    if steep:
                        cells_to_add.append(np.array([y - step_y, x]))
                    else:
                        cells_to_add.append(np.array([x, y - step_y]))
                # Check if we went through the left cell along x while climbing
                elif prev_error + error > 0 and y_dist >= 0:
                    if steep:
                        cells_to_add.append(np.array([y, x - step_x]))
                    else:
                        cells_to_add.append(np.array([x - step_x, y]))
                error -= (2 * dx)

            if steep:
                in_bounds = self.in_bounds(np.array([y, x]), world_coords=False)
                if in_bounds:
                    cells_to_add.append(np.array([y, x]))
            else:
                in_bounds = self.in_bounds(np.array([x, y]), world_coords=False)
                if in_bounds:
                    cells_to_add.append(np.array([x, y]))

            # Among the cells to add, check if we hit an obstacle
            # We should check the cells in the order they were added, since the
            # they are stored in order of increasing distance
            for cell in cells_to_add:
                if not invert and self.is_available(cell, world_coords=False, check_bounds=True):
                    return self.mapToWorld(cell)
                elif invert and not self.is_available(cell, world_coords=False, check_bounds=True):
                    return self.mapToWorld(cell)

            if not in_bounds:
                return None

            prev_error = error
            error += (2 * dy)

        return None

    def get_available_cells_in_radius(self, pos, radius, invert=False):
        obstacles = []
        distances = []

        if not self.get_width() >= pos[0] >= 0 or \
           not self.get_height() >= pos[1] >= 0:
           return obstacles, distances

        angles = np.arange(0, 2*np.pi + self._d_theta, self._d_theta)
        for angle in angles:
            end = np.array([pos[0] + radius * np.cos(angle),
                            pos[1] + radius * np.sin(angle)])
            obstacle = self.find_available(pos, end, invert)
            if obstacle is not None:
                obstacles.append(obstacle)
                distances.append(np.linalg.norm(obstacle - pos))

        return obstacles, distances

    def can_connect(self, pos1, pos2):
        # Check if we can connect from pos1 to pos2
        # using a straight line
        return self.find_obstacle(pos1, pos2) is None