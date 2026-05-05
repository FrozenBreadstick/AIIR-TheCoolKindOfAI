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
        environment_map=None
    ):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-40, -40] + [0]*36, dtype=np.float32),
            high=np.array([40, 40] + [1]*36, dtype=np.float32),
            shape=(38,),
            dtype=np.float32)
        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        self.reached_goal = False
        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None
        self.goal_object = None
        self.goal = None
        self.obstacle_object = None
        self.obstacle_pos = None
        self.has_obstacle = False
        self.lidar_readings = None
        self.done = False
        self.prev_dist_to_goal = None
        self.rendered_img = None
        self.render_rot_matrix = None
        self.building_array = [] # list to keep track of building objects for resetting and cleanup
        self.environment_map = environment_map
        self.end_zone_buffer = 10
        self.plane = None

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
        for i in range(self._actionRepeat):
          self._p.stepSimulation()
          if self._renders:
            time.sleep(self._timeStep)

          car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
          goal_pos, goal_orn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
          car_ob = self.getExtendedObservation()

          # check collisions with the car to enforce termination for unsafe driving (e.g. going offroad or hitting buildings)
          if(self.collision_detect()):
              self.done = True
              break

          if self._termination():
            self.done = True
            break
          self._envStepCounter += 1

        # Compute reward as L2 change in distance to goal
        # dist_to_goal = math.sqrt(((car_ob[0] - self.goal[0]) ** 2 +
                                  # (car_ob[1] - self.goal[1]) ** 2))
        dist_to_goal = math.sqrt(((car_pos[0] - goal_pos[0]) ** 2 +
                                  (car_pos[1] - goal_pos[1]) ** 2))
                                  
        # Check termination constraints so students can't cheat the physics
        # if self.has_obstacle:
        #     dist_to_obs = math.sqrt((car_pos[0] - self.obstacle_pos[0])**2 + (car_pos[1] - self.obstacle_pos[1])**2)
        #     if dist_to_obs < self.minimum_safe_distance:
        #         self.done = True
                
        if dist_to_goal < 1.5 and not self.reached_goal:
            self.done = True
            self.reached_goal = True
            
        if self.reward_callback is not None:
             # Calculate reward via external student function
             reward = self.reward_callback(
                 car_pos=car_pos, 
                 goal_pos=goal_pos,
                 obstacle_pos=self.obstacle_pos,
                 has_obstacle=self.has_obstacle,
                 lidar_readings=self.lidar_readings, # added this for adding the lidar to the callback
                 prev_dist_to_goal=self.prev_dist_to_goal,
                 dist_to_goal=dist_to_goal,
                 reached_goal=self.reached_goal
             )
        else:
            raise ValueError("No reward_callback provided to SimpleDrivingEnv! You must inject the reward logic.")

        self.prev_dist_to_goal = dist_to_goal

        ob = np.array(car_ob, dtype=np.float32)
        return ob, float(reward), self.done, False, dict()

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)
        # Reload the plane and car
        self.plane = Plane(self._p)
        self._envStepCounter = 0
        

        # Clear any existing buildings
        self.building_array = []

        # select which zone of buildings to spawn based on environment_map parameter
        map_data = np.load(self.environment_map)

        obstacle_boundaries = map_data['metrics']
        obstacle_centres = map_data['centroids']
        map_corners = [map_data['min'], map_data['max']]

        #process buildings
        # for i in range(len(obstacle_boundaries)):
        #     self.make_custom_obstacles(obstacle_boundaries[i])

        #select random x_y position for submap corner 1
        map_width_comp = (map_corners[1][0] - map_corners[0][0]) / 2
        map_height_comp = (map_corners[1][1] - map_corners[0][1]) / 2

        # Set the goal to a random target within the map boundaries
        random_x = self.np_random.uniform(map_corners[0][0], map_corners[0][0] + map_width_comp)
        random_y = self.np_random.uniform(map_corners[0][1], map_corners[0][1] + map_height_comp)

        # filter buildings to only those within the selected submap
        for i in range(len(obstacle_boundaries)):
            building_center = obstacle_centres[i]
            if (building_center[0] >= random_x and building_center[0] <= random_x + map_width_comp and
                building_center[1] >= random_y and building_center[1] <= random_y + map_height_comp):
                self.make_custom_obstacles(obstacle_boundaries[i], random_x, random_y) # spawn building if its centroid is within the selected submap

        # make boudary wall points with a little sticking out at the ends for the goals
        boundary_vertices = [
            (random_x - self.end_zone_buffer, random_y, 0),
            (random_x - self.end_zone_buffer, random_y, 2),
            (random_x + map_width_comp + self.end_zone_buffer, random_y, 0),
            (random_x + map_width_comp + self.end_zone_buffer, random_y, 2),
            (random_x + map_width_comp + self.end_zone_buffer, random_y + map_height_comp, 0),
            (random_x + map_width_comp + self.end_zone_buffer, random_y + map_height_comp, 2),
            (random_x - self.end_zone_buffer, random_y + map_height_comp, 0),
            (random_x - self.end_zone_buffer, random_y + map_height_comp, 2)
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

        # Set the goal to end in the opposite side of the map from the car's starting position
        x = (random_x + map_width_comp + (self.end_zone_buffer / 2))
        y = (random_y + (map_height_comp / 2))
        self.goal = (x, y)
        self.done = False
        self.reached_goal = False

        # Visual element of the goal
        self.goal_object = Goal(self._p, self.goal)

        # set car position to be in the opposite mid point from the goal
        car_x = (-(self.end_zone_buffer / 2))
        car_y = ((map_height_comp / 2))
        self.car = Car(self._p, base_position=[car_x, car_y, 0.5])

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

        self.prev_dist_to_goal = math.sqrt(((car_pos[0] - self.goal[0]) ** 2 +
                                           (car_pos[1] - self.goal[1]) ** 2))
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
        goal_pos, goal_orn = self._p.getBasePositionAndOrientation(self.goal_object.goal)
        
        # lidar stuff
        self.lidar_readings = self.car.get_lidar_readings()

        if self.observation_callback is not None:
            # Calculate observation block via external student function
            return self.observation_callback(
                client=self._p,
                car_pos=car_pos,
                car_orn=car_orn,
                goal_pos=goal_pos,
                goal_orn=goal_orn,
                obstacle_pos=self.obstacle_pos,
                has_obstacle=self.has_obstacle,
                lidar_readings=self.lidar_readings # added this for adding the lidar to the callback
            )
        else:
             raise ValueError("No observation_callback provided to SimpleDrivingEnv! You must inject the observation logic.")

    def _termination(self):
        return self._envStepCounter > 4000

    def close(self):
        self._p.disconnect()

    def make_custom_obstacles(self, obstacle_vertices, random_x=0, random_y=0):

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