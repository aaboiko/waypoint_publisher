import rospy
import numpy as np
import open3d as pcl
from numpy.linalg import inv
import matplotlib.pyplot as plt
import actionlib
import waypoint_publisher.msg as msg
from datetime import datetime
import pickle

z_min = -1.0
z_max = 2.0

def read_cloud(filepath):
    cloud = pcl.io.read_point_cloud(filepath)
    rospy.loginfo('Pointcloud obtained: %s', filepath)
    points_array = np.asarray(cloud.points)
    points_number = points_array.shape[0]
    rospy.loginfo('Number of points: %s', points_number)
    return points_array, points_number


def z_filter(cloud, z_min, z_max):
    filtered = []
    for point in cloud:
        z = point[2]
        if z >= z_min and z <= z_max:
            filtered.append(point)
    return np.array(filtered)


def evaluate_cloud(cloud, cell_size):
    xmax, xmin, ymax, ymin, zmax, zmin = -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf
    for point in cloud:
        x = point[0]
        y = point[1]
        z = point[2]
        xmax = max(xmax, x)
        xmin = min(xmin, x)
        ymax = max(ymax, y)
        ymin = min(ymin, y)
        zmax = max(zmax, z)
        zmin = min(zmin, z)

    rospy.loginfo('xmin: %f, xmax: %f, ymin: %f, ymax: %f, zmin: %f, zmax: %f',xmin,xmax,ymin,ymax,zmin,zmax)
    
    cloud_size_x = xmax - xmin
    cloud_size_y = ymax - ymin

    grid_size_x = int(cloud_size_x / cell_size) + 1
    grid_size_y = int(cloud_size_y / cell_size) + 1

    rospy.loginfo('Grid size: x = %i, y = %i', grid_size_x, grid_size_y)
    return grid_size_x, grid_size_y, xmin, xmax, ymin, ymax


def segmentate_cloud(cloud, size_x, size_y, xmin, xmax, ymin, ymax, cell_size):
    point_grid = dict()
    
    for i in range(size_x):
        for j in range(size_y):
            point_grid.setdefault((i,j), [])

    for point in cloud:
        x = int((point[0] - xmin) / cell_size)
        y = int((point[1] - ymin) / cell_size)
        point_grid[(x,y)].append(point)

    rospy.loginfo('Cloud segmentated')
    return point_grid


def get_plane_inclination(segment):
    #rospy.loginfo('Get plane inclination started')
    n = len(segment)
    big = 10e6

    xmax = -np.inf
    xmin = np.inf
    ymax = -np.inf
    ymin = np.inf
    zmax = -np.inf
    zmin = np.inf

    if n < 3:
        return big, big, 0, big, big

    sign = -1.0
    A = np.zeros((n, 3))
    B = sign * np.ones((n, 1))

    for i in range(n):
        x = segment[i][0]
        y = segment[i][1]
        z = segment[i][2]
        A[i,:] = np.array([x, y, z])
        xmax = max(xmax, x)
        xmin = min(xmin, x)
        ymax = max(ymax, y)
        ymin = min(ymin, y)
        zmax = max(zmax, z)
        zmin = min(zmin, z)

    X = inv(np.dot(A.T, A)) @ A.T @ B

    sigma = 0.0
    if n > 3:
        for i in range(n):
            a = X[0]
            b = X[1]
            c = X[2]
            x = segment[i][0]
            y = segment[i][1]
            z = segment[i][2]
            sigma += ((a*x + b*y + c*z - sign)**2) / n

    x_norm = np.linalg.norm(X)
    X = X / x_norm
    z_range_param = zmax - zmin
    
    if abs(X[2]) <= 10e-3:
        return big, sigma, n, z_range_param, zmax

    inc_cos = abs(X[2])
    inc_param = np.sqrt(1.0 - inc_cos**2) / inc_cos

    '''rospy.loginfo('Get plane inclination finished')
    rospy.loginfo('xmin = '+str(xmin)+' xmax = '+str(xmax))
    rospy.loginfo('ymin = '+str(ymin)+' ymax = '+str(ymax))
    rospy.loginfo('zmin = '+str(zmin)+' zmax = '+str(zmax))'''
    return inc_param, sigma, n, z_range_param, zmax


def cost_function(params, model_path):
    #return 2 * (1.0 / (1.0 + np.exp(-np.dot(koefs, params)))) - 1.0
    model = pickle.load(open(model_path, 'rb'))
    return 1 - int(model.predict(params)[0])


def plot_map(occ_map, start, goal):
  plt.imshow(occ_map.T, cmap=plt.cm.gray, interpolation='none', origin='upper')
  plt.plot([start[0]], [start[1]], 'ro')
  plt.plot([goal[0]], [goal[1]], 'go')
  plt.axis([0, occ_map.shape[0]-1, 0, occ_map.shape[1]-1])
  plt.xlabel('x')
  plt.ylabel('y')


def plot_expanded(expanded, start, goal):
  if np.array_equal(expanded, start) or np.array_equal(expanded, goal):
    return
  plt.plot([expanded[0]], [expanded[1]], 'yo')


def plot_path(path, goal):
  if np.array_equal(path, goal):
    return
  plt.plot([path[0]], [path[1]], 'bo')


def plot_costs(cost):
  plt.figure()
  plt.imshow(cost.T, cmap=plt.cm.gray, interpolation='none', origin='upper')
  plt.axis([0, cost.shape[0]-1, 0, cost.shape[1]-1])
  plt.xlabel('x')
  plt.ylabel('y')


def get_neighborhood(cell, occ_map_shape):
    neighbors = []
    x_min = 0
    y_min = 0
    x_max = occ_map_shape[0]
    y_max = occ_map_shape[1]

    def in_bound(cell):
        x = cell[0]
        y = cell[1]
        return x >= x_min and y >= y_min and x < x_max and y < y_max
    
    deltas = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]
    cs = [cell + i for i in deltas]

    for c in cs:
        if in_bound(c):
            neighbors.append(c)

    return neighbors


def get_edge_cost(parent, child, occ_map):
    edge_cost = 0
    thres = 0.5
    k = 10

    if occ_map[child[0], child[1]] < thres:
        edge_cost = np.linalg.norm(parent - child) + k * occ_map[child[0], child[1]]
    else:
        edge_cost = np.Inf

    return edge_cost


def get_heuristic(cell, goal):
    heuristic = 0
    x1 = cell[0]
    y1 = cell[1]
    x2 = goal[0]
    y2 = goal[1]
    heuristic = abs(x1 - x2) + abs(y1 - y2)
   #heuristic = np.linalg.norm(cell - goal)

    return heuristic


def float_to_grid(point, offset, cell_size):
    x = int((point[0] - offset[0]) / cell_size)
    y = int((point[1] - offset[1]) / cell_size)
    return [x, y]


def grid_to_float(point, offset, cell_size):
    x = cell_size * point[0] + cell_size / 2 + offset[0]
    y = cell_size * point[1] + cell_size / 2 + offset[1]
    return [x, y]


def run_path_planning(occ_map, initial, dest, offset, cell_size, params):
    rospy.loginfo('Path planning in progress...')

    costs = np.ones(occ_map.shape) * np.inf
    closed_flags = np.zeros(occ_map.shape)
    predecessors = -np.ones(occ_map.shape + (2,), dtype=np.int)
    path_array = []
    success = False

    start = float_to_grid(initial, offset, cell_size)
    parent = start
    goal = float_to_grid(dest, offset, cell_size)
    costs[start[0], start[1]] = 0

    heuristic = np.zeros(occ_map.shape)
    for x in range(occ_map.shape[0]):
        for y in range(occ_map.shape[1]):
            heuristic[x, y] = get_heuristic([x, y], goal)

    while not np.array_equal(parent, goal):
        open_costs = np.where(closed_flags==1, np.inf, costs) + heuristic
        x, y = np.unravel_index(open_costs.argmin(), open_costs.shape)

        if open_costs[x, y] == np.inf:
            break

        parent = np.array([x, y])
        closed_flags[x, y] = 1
        #rospy.loginfo('Current parent cell: ' + str(parent))

        neighs = get_neighborhood(parent, occ_map.shape)
        #rospy.loginfo('neighs: ' + str(neighs))
        for neigh in neighs:
            #rospy.loginfo('Current neigh: ' + str(neigh))
            if costs[neigh[0], neigh[1]] > costs[x, y] + get_edge_cost(parent, neigh, occ_map):
                costs[neigh[0], neigh[1]] = costs[x, y] + get_edge_cost(parent, neigh, occ_map)
                predecessors[neigh[0], neigh[1]] = parent

    if np.array_equal(parent, goal):
        #rospy.loginfo('Parent and goal are equal')
        path_length = 0

        while predecessors[parent[0], parent[1]][0] >= 0:
            predecessor = predecessors[parent[0], parent[1]]
            path_length += np.linalg.norm(parent - predecessor)
            path_array.insert(0, parent)
            parent = predecessor

        success = True
        rospy.loginfo( "found goal     : " + str(parent) )
        rospy.loginfo( "cells expanded : " + str(np.count_nonzero(closed_flags)))
        rospy.loginfo( "path cost      : " + str(costs[goal[0], goal[1]]))
        rospy.loginfo( "path length    : " + str(path_length))
    else:
        rospy.loginfo( "no valid path found")

    return path_array, success


def get_costmap(point_grid, size_x, size_y, koefs, model_path):
    costmap = np.zeros((size_x, size_y))
    rospy.loginfo('Creating costmap...')
    iter = 0.0
    progress = 0
    prev = 0
    count = size_x * size_y

    for i in range(size_x):
        for j in range(size_y):
            inc_param, sigma, n, z_range_param, zmax = get_plane_inclination(point_grid[(i,j)])
            params = np.array([[float(inc_param), float(sigma), float(z_range_param)]])
            costmap[i,j] = cost_function(params, model_path)
            iter += 1.0
            progress = int((iter / count) * 100)
            
            if progress != prev:
                rospy.loginfo('Costmap generating in progress: [' + str(progress) + '%]')
                prev = progress

    rospy.loginfo('Costmap generated')
    return costmap


class GetPathAction(object):
    _feedback = msg.GetPathFeedback()
    _result = msg.GetPathResult()

    def __init__(self, name, params):
        self._action_name = name
        self._as = actionlib.SimpleActionServer(self._action_name, msg.GetPathAction, execute_cb=self.execute_cb, auto_start = False)
        self._as.start()
        self.params = params

    def execute_cb(self, goal):
        save_map = self.params[3]
        filepath = self.params[4]
        model_path = self.params[5]

        cell_size = float(goal.cell_size)
        koefs = np.array(goal.koefs)
        start = [float(goal.start.x), float(goal.start.y)]
        dest = [float(goal.destination.x), float(goal.destination.y)]
        offset = [float(goal.offset.x), float(goal.offset.y)]

        cloud, points_number = read_cloud(filepath)
        grid_size_x, grid_size_y, xmin, xmax, ymin, ymax = evaluate_cloud(cloud, cell_size)
        offset = [xmin, ymin]
        point_grid = segmentate_cloud(cloud, grid_size_x, grid_size_y, xmin, xmax, ymin, ymax, cell_size)
        costmap = get_costmap(point_grid, grid_size_x, grid_size_y, koefs, model_path)
        path_array, success = run_path_planning(costmap, start, dest, offset, cell_size, self.params)

        if save_map:
            filename = 'src/waypoint_publisher/maps/map_'
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y_%H:%M:%S")
            filename += dt_string
            filename += '.txt'
            np.savetxt(filename, costmap, delimiter=' ')
            rospy.loginfo('Costmap saved to ' + filename)
        
        res_array = []
        for point in path_array:
            point_f = grid_to_float(point, offset, cell_size)
            point2f = msg.Point2f(x=point_f[0], y=point_f[1])
            res_array.append(point2f)

        self._result.path = res_array
        self._result.success = success
        self._result.xmin = xmin
        self._result.ymin = ymin
        self._result.xmax = xmax
        self._result.ymax = ymax

        self._as.set_succeeded(self._result)


def run_node():
    rospy.init_node('pointcloud_processor')
    plot_map_param = rospy.get_param('plot_map', False)
    plot_expanded_param = rospy.get_param('plot_expanded', False)
    plot_costs_param = rospy.get_param('plot_costs', False)
    save_map = rospy.get_param('save_map', True)
    filepath = rospy.get_param('filepath', '/home/anatoliy/.ros/cloud_lab_1.ply')
    model_path = rospy.get_param('model_path', 'src/waypoint_publisher/models/house_1.sav')

    params = [
        plot_map_param, 
        plot_expanded_param, 
        plot_costs_param, 
        save_map, 
        filepath,
        model_path
    ]
    rospy.loginfo('Node pointcloud_processor init')

    server = GetPathAction(rospy.get_name(), params)
    rospy.spin()

try:
    run_node()
except rospy.ROSInterruptException:
    rospy.loginfo('Unexpected ROSInterruptException')