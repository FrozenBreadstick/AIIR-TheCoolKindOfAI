import gymnasium as gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.goal import Goal
from simple_driving.resources.obstacle import Obstacle
import matplotlib.pyplot as plt
import time

# astar imports
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class SimpleDrivingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'fp_camera', 'tp_camera', 'rgb_array']}

    def __init__(
        self, 
        isDiscrete=True, 
        renders=False, 
        minimum_safe_distance=1.0,
        reward_callback=None,
        observation_callback=None,
        environment_map=None,
        checkpoint_frequency=10,
        **kwargs
    ):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-600, -600, -600, -600] + [0.0]*36, dtype=np.float32),
            high=np.array([600, 600, 600, 600] + [1.0]*36, dtype=np.float32),
            shape=(40,),
            dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal_1 = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object_1 = None
        self.goal_1 = None
        self.checkpoint_array = [] # list to keep track of checkpoint goal objects for resetting and cleanup
        self.obstacle_object = None
        self.obstacle_pos = None
        self.has_obstacle = False
        self.lidar_readings = None
        self.done = False
        self.prev_dist_to_goal_1 = None
        self.prev_dist_to_checkpoint = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.building_array = [] # list to keep track of building objects for resetting and cleanup
        self.environment_map = environment_map
        self.end_zone_buffer = 10
        self.plane = None
        self.map_height = 250
        self.map_width = 250
        self.astar_grid = np.ones((self.map_width, self.map_height), dtype=int)  # will be size map_width x map_height, with 1 for free space and 0 for occupied by building, used for pathfinding in the reward shaping
        self.step_counter = 0
        self.collision_detected = False
        self.goal_1_reward_given = False
        self.checkpoint_frequency = checkpoint_frequency # how many path points to skip before spawning a checkpoint, can be tuned based on the density of the map and the desired difficulty of the task

        # --- Configurable Limits ---
        self.minimum_safe_distance = minimum_safe_distance
        
        # Callbacks for Student Assignment
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        # ------------------------------------------------
        
        self._envStepCounter = 0

    def step(self, action):
        # Feed action to the car and get observation of car's state
        if (self._isDiscrete):
            fwd = [-1, -1, -1, 0, 0, 0, 1, 1, 1]
            steerings = [-0.6, 0, 0.6, -0.6, 0, 0.6, -0.6, 0, 0.6]
            throttle = fwd[action]
            steering_angle = steerings[action]
            action = [throttle, steering_angle]
        self.car.apply_action(action)
        self.last_steering = action[1]
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
          goal_pos_1, goal_orn_1 = self._p.getBasePositionAndOrientation(self.goal_object_1.goal)
          checkpoint_pos = self.checkpoint_array[0].position if len(self.checkpoint_array) > 0 else None
          car_ob = self.getExtendedObservation()

          # check collisions with the car to enforce termination for unsafe driving (e.g. going offroad or hitting buildings)
          if(self.collision_detect()):
              self.done = True
              self.collision_detected = True
              break

          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  # (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal_1 = math.sqrt(((car_pos[0] - goal_pos_1[0]) ** 2 +
                                  (car_pos[1] - goal_pos_1[1]) ** 2))
                                  
        # Check termination constraints so students can't cheat the physics
        # if self.has_obstacle:
        #     dist_to_obs = math.sqrt((car_pos[0] - self.obstacle_pos[0])**2 + (car_pos[1] - self.obstacle_pos[1])**2)
        #     if dist_to_obs < self.minimum_safe_distance:
        #         self.done = True
                
        if dist_to_goal_1 < 1.5 and not self.goal_1_reward_given:
            self.reached_goal_1 = True
            self.goal_1_reward_given = True

        # if dist_to_goal_2 < 1.5 and not self.goal_2_reward_given:
        #     self.reached_goal_2 = True
        #     self.goal_2_reward_given = True

        # if dist_to_goal_3 < 1.5 and not self.goal_3_reward_given:
        #     self.reached_goal_3 = True
        #     self.goal_3_reward_given = True

        checkpoint_reached = False

        if len(self.checkpoint_array) > 0:
            dist_to_checkpoint = math.sqrt(((car_pos[0] - checkpoint_pos[0]) ** 2 +
                                (car_pos[1] - checkpoint_pos[1]) ** 2))
            if dist_to_checkpoint < 1.5:
                self.checkpoint_array.pop(0) # remove checkpoint once reached to encourage the car to follow the path
                checkpoint_reached = True
        else:
            dist_to_checkpoint = None

        if self.reward_callback is not None:
             # Calculate reward via external student function
             reward = self.reward_callback(
                 car_pos=car_pos, 
                 goal_pos_1=goal_pos_1,
                 checkpoint_pos=checkpoint_pos,
                 lidar_readings=self.lidar_readings, # added this for adding the lidar to the callback
                 dist_to_goal_1=dist_to_goal_1,
                 dist_to_checkpoint=dist_to_checkpoint,
                 prev_dist_to_goal_1=self.prev_dist_to_goal_1,
                 prev_dist_to_checkpoint=self.prev_dist_to_checkpoint if len(self.checkpoint_array) > 0 else None,
                 reached_goal_1=self.reached_goal_1,
                 reached_checkpoint=checkpoint_reached,
                 collided=self.collision_detected
             )
        else:
            raise ValueError("No reward_callback provided to SimpleDrivingEnv! You must inject the reward logic.")

        self.prev_dist_to_goal_1 = dist_to_goal_1
        if dist_to_checkpoint is not None:
            self.prev_dist_to_checkpoint = dist_to_checkpoint
        
        if self.goal_1_reward_given:
            self.done = True

        ob = np.array(car_ob, dtype=np.float32)

        # if self.step_counter % 1000 == 0:
        #     print("obs at step", self.step_counter, ": ", ob) # debug print to check initial observation

        # centre camera on the car for testing
        car_id = self.car.get_ids()
        car_pos, _ = self._p.getBasePositionAndOrientation(car_id)
        # raise camera spawn height
        camera_pos = [car_pos[0], car_pos[1], 5]

        self._p.resetDebugVisualizerCamera(
            cameraDistance=8,          # zoom out enough to see car
            cameraYaw=-90,               # rotation around car
            cameraPitch=-45,           # look slightly down
            cameraTargetPosition=camera_pos
        )

        self.step_counter += 1

        return ob, float(reward), self.done, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        self.step_counter = 0
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        self.plane = Plane(self._p)
        self._envStepCounter = 0
        self.step_count = 0
        self.collision_detected = False
        self.done = False
        self.reached_goal_1 = False
        self.prev_dist_to_goal_1 = None
        self.goal_1_reward_given = False
        self.checkpoint_array = [] # clear checkpoints
        self.astar_grid = np.ones((self.map_width, self.map_height), dtype=int)  # reset astar grid
        self.prev_dist_to_checkpoint = None

        # Clear any existing buildings
        self.building_array = []

        # select which zone of buildings to spawn based on environment_map parameter
        map_data = np.load(self.environment_map)

        obstacle_boundaries = map_data['metrics']
        obstacle_centres = map_data['centroids']
        map_corners = [map_data['min'], map_data['max']]

        

        #select random x_y position for submap corner 1
        map_width_comp = (map_corners[1][0] - map_corners[0][0]) / 2
        map_height_comp = (map_corners[1][1] - map_corners[0][1]) / 2

        #print("map height/width comp:", map_height_comp*2, map_width_comp*2)

        if(map_width_comp >= 0):
            boundary_width = self.map_width
        else:
            boundary_width = -self.map_width

        if(map_height_comp >= 0):
            boundary_height = self.map_height
        else:
            boundary_height = -self.map_height

        # Set the goal to a random target within the map boundaries
        random_x = self.np_random.uniform(map_corners[0][0], map_corners[0][0] + boundary_width)
        random_y = self.np_random.uniform(map_corners[0][1], map_corners[0][1] + boundary_height)

        # Set the goal1 to end in the opposite side of the map from the car's starting position
        x1 = (boundary_width - self.end_zone_buffer)
        y1 = ((boundary_height / 2))
        self.goal_1 = (x1, y1)
        
        # Visual element of the goal
        self.goal_object_1 = Goal(self._p, self.goal_1)

        # set car position to be in the opposite mid point from the goal
        car_x = ((self.end_zone_buffer / 2))
        car_y = ((boundary_height / 2))
        self.car = Car(self._p, base_position=[car_x, car_y, 0.5])

        # # Set the goal2 to somewhere in the middle of the map from the car's starting position
        # left_or_right_goal_midpoint = self.np_random.choice([True, False])
        # if left_or_right_goal_midpoint: 
        #     multiplier = 1
        # else: 
        #     multiplier = 3 
        # x2 = (boundary_width / 2)
        # y2 = ((boundary_height / 4) * multiplier)
        # self.goal_2 = (x2, y2)
        # #visual element of the goal
        # self.goal_object_2 = Goal(self._p, self.goal_2)

        # # Set the goal3 to somewhere in the front of the map near the car's starting position
        # left_or_right_goal_front = self.np_random.choice([True, False])
        # if left_or_right_goal_front:
        #     multiplier = 1
        # else:
        #     multiplier = 3
        # x3 = (boundary_width / 4)
        # y3 = ((boundary_height / 4) * multiplier)
        # self.goal_3 = (x3, y3)
        # #visual element of the goal
        # self.goal_object_3 = Goal(self._p, self.goal_3)

        # filter buildings to only those within the selected submap and not around the goals (to ensure at least some feasible paths to the goals and to prevent impossible scenarios where the car starts in a building)
        for i in range(len(obstacle_boundaries)):
            building_center = obstacle_centres[i]
            if (building_center[0] >= random_x and building_center[0] <= random_x + boundary_width and
                building_center[1] >= random_y and building_center[1] <= random_y + boundary_height):
                self.make_custom_obstacles(obstacle_boundaries[i], random_x, random_y) # spawn building if its centroid is within the selected submap

        grid = Grid(matrix=self.astar_grid.tolist())
        start = grid.node(int(self.end_zone_buffer/2), int(boundary_height/2)) # start is the car's initial position
        # print("goal 1 position:", self.goal_1)
        # print(int(x1), int(y1))
        # print("grid width =", len(grid.nodes[0]))
        # print("grid height =", len(grid.nodes))
        # print("goal world =", x1, y1)
        end = grid.node(int(x1), int(y1)) # end is the goal position
        finder = AStarFinder()
        path, runs = finder.find_path(start, end, grid)

        # print("A* path length:", len(path))
        # if len(path) == 0:
        #     print("WARNING: No path found → no checkpoints will spawn")

        for i in range(len(path)-1):
            if i % self.checkpoint_frequency == 0 and i != 0: # only spawn obstacles on some of the path points to ensure there are gaps to drive through
                path_x = path[i].x
                path_y = path[i].y
                self.checkpoint_array.append(Goal(self._p, (path_x, path_y))) # spawn checkpoint goals along the path to encourage the car to follow the path

        # make boudary wall points with a little sticking out at the ends for the goals
        boundary_vertices = [
            (random_x - self.end_zone_buffer, random_y, 0),
            (random_x - self.end_zone_buffer, random_y, 2),
            (random_x + boundary_width + self.end_zone_buffer, random_y, 0),
            (random_x + boundary_width + self.end_zone_buffer, random_y, 2),
            (random_x + boundary_width + self.end_zone_buffer, random_y + boundary_height, 0),
            (random_x + boundary_width + self.end_zone_buffer, random_y + boundary_height, 2),
            (random_x - self.end_zone_buffer, random_y + boundary_height, 0),
            (random_x - self.end_zone_buffer, random_y + boundary_height, 2)
        ]

        boundary_indices = [
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6],
            [5, 6, 7],
            [6, 7, 0],
            [7, 0, 1]
        ]
        boundary_indices = np.array(boundary_indices).astype(np.int32).flatten().tolist()
        

        # create collision and visual shapes for the boundary walls
        col_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_MESH, vertices=boundary_vertices, indices=boundary_indices)
        vis_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH, vertices=boundary_vertices, indices=boundary_indices, rgbaColor=[1, 0, 0, 1]) # Red color

        # create the multi body for the custom obstacle
        obstacle_object = self._p.createMultiBody(
            baseMass=0, # Infinite mass, completely static
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[-random_x, -random_y, 0] # position is irrelevant since vertices are in world coordinates
        )

        # Obstacle logic
        # scenario = options.get("scenario", "random") if options else "random"
        # if scenario == "none":
        #     self.has_obstacle = False
        # elif scenario == "midpoint":
        #     self.has_obstacle = True
        #     force_midpoint = True
        # elif scenario == "random_pos":
        #     self.has_obstacle = True
        #     force_midpoint = False
        # else: # random
        #     self.has_obstacle = self.np_random.random() < 0.60
        #     force_midpoint = self.np_random.random() < 0.5
            
        # if self.has_obstacle:
        #     if force_midpoint:
        #         # Midpoint
        #         obs_x = self.goal[0] / 2.0
        #         obs_y = self.goal[1] / 2.0
        #     else:
        #         # Random position with min distance 1.5 from origin and goal
        #         while True:
        #             obs_x = self.np_random.uniform(-9, 9)
        #             obs_y = self.np_random.uniform(-9, 9)
        #             dist_to_origin = math.sqrt(obs_x**2 + obs_y**2)
        #             dist_to_goal_pt = math.sqrt((obs_x - self.goal[0])**2 + (obs_y - self.goal[1])**2)
        #             if dist_to_origin > 1.5 and dist_to_goal_pt > 1.5:
        #                 break
        #     self.obstacle_pos = (obs_x, obs_y)
        #     self.obstacle_object = Obstacle(self._p, self.obstacle_pos)
        # else:
        #     self.obstacle_pos = None
        #     self.obstacle_object = None

        # Get observation to return
        car_pos = self.car.get_observation()

        self.prev_dist_to_goal_1 = math.sqrt(((car_pos[0] - self.goal_1[0]) ** 2 +
                                             (car_pos[1] - self.goal_1[1]) ** 2))
        
        checkpoint_pos = self.checkpoint_array[0].position if len(self.checkpoint_array) > 0 else None
        self.prev_dist_to_checkpoint = math.sqrt(((car_pos[0] - checkpoint_pos[0]) ** 2 +
                                                 (car_pos[1] - checkpoint_pos[1]) ** 2)) if checkpoint_pos is not None else None

        car_ob = self.getExtendedObservation()

        # centre camera on the car for testing
        car_id = self.car.get_ids()
        car_pos, _ = self._p.getBasePositionAndOrientation(car_id)
        # raise camera spawn height
        camera_pos = [car_pos[0], car_pos[1], 5]

        self._p.resetDebugVisualizerCamera(
            cameraDistance=8,          # zoom out enough to see car
            cameraYaw=0,               # rotation around car
            cameraPitch=-80,           # look slightly down
            cameraTargetPosition=camera_pos
        )

        return np.array(car_ob, dtype=np.float32), dict()

    def render(self, mode='human'):
        if mode == "fp_camera":
            # Base information
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1,
                                                       nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in
                        self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2

            # Rotate camera direction
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)

            # Display image
            # frame = self._p.getCameraImage(100, 100, view_matrix, proj_matrix)[2]
            # frame = np.reshape(frame, (100, 100, 4))
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                      height=RENDER_HEIGHT,
                                                      viewMatrix=view_matrix,
                                                      projectionMatrix=proj_matrix,
                                                      renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
            # self.rendered_img.set_data(frame)
            # plt.draw()
            # plt.pause(.00001)

        elif mode == "tp_camera":
            car_id = self.car.get_ids()
            base_pos, orn = self._p.getBasePositionAndOrientation(car_id)
            view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                                    distance=20.0,
                                                                    yaw=40.0,
                                                                    pitch=-35,
                                                                    roll=0,
                                                                    upAxisIndex=2)
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                             aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                             nearVal=0.1,
                                                             farVal=100.0)
            # (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
            #                                           height=RENDER_HEIGHT,
            #                                           viewMatrix=view_matrix,
            #                                           projectionMatrix=proj_matrix,
            #                                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)
            frame = frame[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        goal_pos_1, goal_orn_1 = self._p.getBasePositionAndOrientation(self.goal_object_1.goal)
        checkpoint_pos = self.checkpoint_array[0].position if len(self.checkpoint_array) > 0 else None
        checkpoint_orn = (0, 0, 0, 1) if checkpoint_pos is not None else None

        # lidar stuff
        self.lidar_readings = self.car.get_lidar_readings()

        if self.observation_callback is not None:
            # Calculate observation block via external student function
            return self.observation_callback(
                client=self._p,
                car_pos=car_pos,
                car_orn=car_orn,
                goal_pos_1=goal_pos_1,
                goal_orn_1=goal_orn_1,
                checkpoint_pos=checkpoint_pos,
                checkpoint_orn=checkpoint_orn,
                lidar_readings=self.lidar_readings # added this for adding the lidar to the callback
            )
        else:
             raise ValueError("No observation_callback provided to SimpleDrivingEnv! You must inject the observation logic.")

    def _termination(self):
        return self._envStepCounter > 50000
        return self._envStepCounter > 50000

    def close(self):
        self._p.disconnect()

    def make_custom_obstacles(self, obstacle_vertices, random_x=0, random_y=0):

        #check a goal is not within the building polygon, if it is, skip spawning that building to ensure there is always a feasible path to the goals
        for goal in [self.goal_1]:
            if self.point_in_convex_quad(goal, obstacle_vertices, random_x, random_y):
                return
            
        for car in [self.car]:
            pos = self._p.getBasePositionAndOrientation(car.get_ids())[0]
            # print("car pos:", pos[:2])
            if self.point_in_convex_quad(pos[:2], obstacle_vertices, random_x, random_y):
                return
            
        # update the astar grid to mark building locations as unwalkable
        for i in range(self.map_width):
            for j in range(self.map_height):
                if self.point_in_convex_quad((i, j), obstacle_vertices, random_x, random_y):
                    self.astar_grid[j][i] = 0 # mark as unwalkable (note the indexing order for y,x due to row-major order of the grid)
                     # --- INFLATION (buffer zone around obstacles) ---
                    INFLATION_RADIUS = 3  # try 1–3

                    for dx in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                        for dy in range(-INFLATION_RADIUS, INFLATION_RADIUS + 1):
                            ni = i + dx
                            nj = j + dy

                            if 0 <= ni < self.map_width and 0 <= nj < self.map_height:
                                self.astar_grid[nj][ni] = 0

        # change the vertices shape from list of 4 (x,y) to list of 8 (x,y,z) for pybullet
        obstacle_vertices_3d = []
        for vertex in obstacle_vertices:
            obstacle_vertices_3d.append((vertex[0], vertex[1], 0.0)) # add z=0 for all vertices on the ground plane
            obstacle_vertices_3d.append((vertex[0], vertex[1], 1.0)) # add z=1 for all vertices to make it a vertical wall

        # make a index list for the mesh (2 triangles per quad) 
        indices = []
        indices.append([0, 1, 2])
        indices.append([1, 2, 3])
        indices.append([2, 3, 4])
        indices.append([3, 4, 5])
        indices.append([4, 5, 6])
        indices.append([5, 6, 7])
        indices.append([6, 7, 0])
        indices.append([7, 0, 1])   # mannually because im too lazy to write a loop for this. and dont need top or bottom
        indices = np.array(indices).astype(np.int32).flatten().tolist()

        # debugging
        # print("Vertices shape:", np.array(obstacle_vertices_3d).shape)
        # print("Indices sample:", np.array(indices)[:10])
        # print("Indices type:", type(np.array(indices)[0]))    


        # create collision and visual shapes for the custom obstacles
        col_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_MESH, vertices=obstacle_vertices_3d, indices=indices)
        vis_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH, vertices=obstacle_vertices_3d, indices=indices, rgbaColor=[1, 0, 0, 1]) # Red color

        # create the multi body for the custom obstacle
        obstacle_object = self._p.createMultiBody(
            baseMass=0, # Infinite mass, completely static
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[-random_x, -random_y, 0] # position is irrelevant since vertices are in world coordinates
        )

        self.building_array.append(obstacle_object)


    # exclusively to stop collisions with the ground plane from being counted as collisions with the environment for termination purposes. Only call this after stepping the simulation and before checking for termination
    def collision_detect(self):
        for c in self._p.getContactPoints(bodyA=self.car.car):
            if c[2] != self.plane.get_ids(): # if the car is colliding with anything other than the ground plane, return True for collision
                return True
        return False
    
    # checks if a goal is within a building by checking if the goal point is within the polygon defined by the building vertices. This is used to filter out buildings that are on top of the goals when spawning the environment, to ensure there is always a feasible path to the goals
    def point_in_convex_quad(self, point, polygon, random_x, random_y):
        x, y = point

        def cross(ax, ay, bx, by):
            return ax * by - ay * bx

        sign = None
        n = len(polygon)

        for i in range(n):
            x1 = polygon[i][0] - random_x
            y1 = polygon[i][1] - random_y
            x2 = polygon[(i + 1) % n][0] - random_x
            y2 = polygon[(i + 1) % n][1] - random_y

            edge_x = x2 - x1
            edge_y = y2 - y1

            point_x = x - x1
            point_y = y - y1

            c = cross(edge_x, edge_y, point_x, point_y)

            if c != 0:
                if sign is None:
                    sign = c > 0
                elif (c > 0) != sign:
                    return False

        return True