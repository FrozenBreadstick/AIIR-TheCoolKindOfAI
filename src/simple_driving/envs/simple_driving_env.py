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
        environment_map=None,
        max_steps=10000
    ):
        if (isDiscrete):
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.action_space = gym.spaces.box.Box(
                low=np.array([-1, -.6], dtype=np.float32),
                high=np.array([1, .6], dtype=np.float32))

        # Observation space: 6 (3 goals x,y) + 36 lidar = 42
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-2000, -2000, -2000, -2000, -2000, -2000] + [0]*36, dtype=np.float32),
            high=np.array([2000, 2000, 2000, 2000, 2000, 2000] + [1]*36, dtype=np.float32),
            shape=(42,),
            dtype=np.float32)

        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
          self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
          self._p = bc.BulletClient()

        # Single goal flags replaced by 3 goal flags
        self.reached_goal_1 = False
        self.reached_goal_2 = False
        self.reached_goal_3 = False
        self.goal_1_reward_given = False
        self.goal_2_reward_given = False
        self.goal_3_reward_given = False

        self._timeStep = 0.01
        self._actionRepeat = 50
        self._renders = renders
        self._isDiscrete = isDiscrete
        self.car = None

        # 3 goal objects
        self.goal_object_1 = None
        self.goal_object_2 = None
        self.goal_object_3 = None
        self.goal_1 = None
        self.goal_2 = None
        self.goal_3 = None

        self.obstacle_object = None
        self.obstacle_pos = None
        self.has_obstacle = False
        self.lidar_readings = None
        self.done = False

        # 3 prev distances
        self.prev_dist_to_goal_1 = None
        self.prev_dist_to_goal_2 = None
        self.prev_dist_to_goal_3 = None

        self.rendered_img = None
        self.render_rot_matrix = None
        self.building_array = []
        self.environment_map = environment_map
        self.end_zone_buffer = 10
        self.plane = None
        self.map_height = 250
        self.map_width = 250
        self.step_counter = 0
        self.collision_detected = False
        self.last_steering = 0.0
        self.max_steps = max_steps

        # --- Configurable Limits ---
        self.minimum_safe_distance = minimum_safe_distance
        
        # Callbacks
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        
        self._envStepCounter = 0

    def step(self, action):
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
            goal_pos_2, goal_orn_2 = self._p.getBasePositionAndOrientation(self.goal_object_2.goal)
            goal_pos_3, goal_orn_3 = self._p.getBasePositionAndOrientation(self.goal_object_3.goal)
            car_ob = self.getExtendedObservation()

            if self.collision_detect():
                self.done = True
                self.collision_detected = True
                break

            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        # Distances to all 3 goals
        dist_to_goal_1 = math.sqrt((car_pos[0] - goal_pos_1[0])**2 + (car_pos[1] - goal_pos_1[1])**2)
        dist_to_goal_2 = math.sqrt((car_pos[0] - goal_pos_2[0])**2 + (car_pos[1] - goal_pos_2[1])**2)
        dist_to_goal_3 = math.sqrt((car_pos[0] - goal_pos_3[0])**2 + (car_pos[1] - goal_pos_3[1])**2)

        # Check if each goal is reached (only once each)
        if dist_to_goal_1 < 1.5 and not self.goal_1_reward_given:
            self.reached_goal_1 = True
            self.goal_1_reward_given = True

        if dist_to_goal_2 < 1.5 and not self.goal_2_reward_given:
            self.reached_goal_2 = True
            self.goal_2_reward_given = True

        if dist_to_goal_3 < 1.5 and not self.goal_3_reward_given:
            self.reached_goal_3 = True
            self.goal_3_reward_given = True

        # End episode only when all 3 goals are reached
        if self.goal_1_reward_given and self.goal_2_reward_given and self.goal_3_reward_given:
            self.done = True

        if self.reward_callback is not None:
            reward = self.reward_callback(
                car_pos=car_pos,
                goal_pos_1=goal_pos_1,
                goal_pos_2=goal_pos_2,
                goal_pos_3=goal_pos_3,
                lidar_readings=self.lidar_readings,
                prev_dist_to_goal_1=self.prev_dist_to_goal_1,
                prev_dist_to_goal_2=self.prev_dist_to_goal_2,
                prev_dist_to_goal_3=self.prev_dist_to_goal_3,
                dist_to_goal_1=dist_to_goal_1,
                dist_to_goal_2=dist_to_goal_2,
                dist_to_goal_3=dist_to_goal_3,
                reached_goal_1=self.reached_goal_1,
                reached_goal_2=self.reached_goal_2,
                reached_goal_3=self.reached_goal_3,
                goal_1_reward_given=self.goal_1_reward_given,
                goal_2_reward_given=self.goal_2_reward_given,
                goal_3_reward_given=self.goal_3_reward_given,
                collided=self.collision_detected
            )
        else:
            raise ValueError("No reward_callback provided!")

        # Reset per-step flags (keep reward_given flags — they persist through episode)
        self.reached_goal_1 = False
        self.reached_goal_2 = False
        self.reached_goal_3 = False

        # Update previous distances
        self.prev_dist_to_goal_1 = dist_to_goal_1
        self.prev_dist_to_goal_2 = dist_to_goal_2
        self.prev_dist_to_goal_3 = dist_to_goal_3

        ob = np.array(car_ob, dtype=np.float32)
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

        self.plane = Plane(self._p)
        self._envStepCounter = 0
        self.step_count = 0
        self.collision_detected = False
        self.done = False

        # Reset all goal flags
        self.reached_goal_1 = False
        self.reached_goal_2 = False
        self.reached_goal_3 = False
        self.goal_1_reward_given = False
        self.goal_2_reward_given = False
        self.goal_3_reward_given = False

        # Clear buildings
        self.building_array = []

        # Load map
        map_data = np.load(self.environment_map)
        obstacle_boundaries = map_data['metrics']
        obstacle_centres = map_data['centroids']
        map_corners = [map_data['min'], map_data['max']]

        map_width_comp = (map_corners[1][0] - map_corners[0][0]) / 2
        map_height_comp = (map_corners[1][1] - map_corners[0][1]) / 2

        boundary_width = self.map_width if map_width_comp >= 0 else -self.map_width
        boundary_height = self.map_height if map_height_comp >= 0 else -self.map_height

        random_x = self.np_random.uniform(map_corners[0][0], map_corners[0][0] + boundary_width)
        random_y = self.np_random.uniform(map_corners[0][1], map_corners[0][1] + boundary_height)

        # Spawn buildings in submap
        for i in range(len(obstacle_boundaries)):
            building_center = obstacle_centres[i]
            if (building_center[0] >= random_x and building_center[0] <= random_x + boundary_width and
                building_center[1] >= random_y and building_center[1] <= random_y + boundary_height):
                self.make_custom_obstacles(obstacle_boundaries[i], random_x, random_y)

        # Boundary walls
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
        boundary_indices = np.array([
            [0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,0],[7,0,1]
        ]).astype(np.int32).flatten().tolist()

        col_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_MESH, vertices=boundary_vertices, indices=boundary_indices)
        vis_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH, vertices=boundary_vertices, indices=boundary_indices, rgbaColor=[1, 0, 0, 1])
        self._p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col_shape_id, baseVisualShapeIndex=vis_shape_id, basePosition=[-random_x, -random_y, 0])

        # --- Spawn 3 goals at different distances ---
        # Goal 1: close (1/4 of the way across)
        x1 = boundary_width * 0.25
        y1 = boundary_height / 2
        self.goal_1 = (x1, y1)
        self.goal_object_1 = Goal(self._p, self.goal_1)

        # Goal 2: middle (1/2 of the way across)
        x2 = boundary_width * 0.5
        y2 = boundary_height / 2
        self.goal_2 = (x2, y2)
        self.goal_object_2 = Goal(self._p, self.goal_2)

        # Goal 3: far (end of the map)
        x3 = boundary_width * 0.75
        y3 = boundary_height / 2
        self.goal_3 = (x3, y3)
        self.goal_object_3 = Goal(self._p, self.goal_3)

        # Car starts at the opposite side
        car_x = -(self.end_zone_buffer / 2)
        car_y = boundary_height / 2
        self.car = Car(self._p, base_position=[car_x, car_y, 0.5])

        # Initial distances
        car_pos = self.car.get_observation()
        self.prev_dist_to_goal_1 = math.sqrt((car_pos[0] - self.goal_1[0])**2 + (car_pos[1] - self.goal_1[1])**2)
        self.prev_dist_to_goal_2 = math.sqrt((car_pos[0] - self.goal_2[0])**2 + (car_pos[1] - self.goal_2[1])**2)
        self.prev_dist_to_goal_3 = math.sqrt((car_pos[0] - self.goal_3[0])**2 + (car_pos[1] - self.goal_3[1])**2)

        car_ob = self.getExtendedObservation()

        # Camera
        car_id = self.car.get_ids()
        car_pos, _ = self._p.getBasePositionAndOrientation(car_id)
        self._p.resetDebugVisualizerCamera(
            cameraDistance=50,
            cameraYaw=0,
            cameraPitch=-80,
            cameraTargetPosition=[car_pos[0], car_pos[1], 0]
        )

        return np.array(car_ob, dtype=np.float32), dict()

    def render(self, mode='human'):
        if mode == "fp_camera":
            car_id = self.car.get_ids()
            proj_matrix = self._p.computeProjectionMatrixFOV(fov=80, aspect=1, nearVal=0.01, farVal=100)
            pos, ori = [list(l) for l in self._p.getBasePositionAndOrientation(car_id)]
            pos[2] = 0.2
            rot_mat = np.array(self._p.getMatrixFromQuaternion(ori)).reshape(3, 3)
            camera_vec = np.matmul(rot_mat, [1, 0, 0])
            up_vec = np.matmul(rot_mat, np.array([0, 0, 1]))
            view_matrix = self._p.computeViewMatrix(pos, pos + camera_vec, up_vec)
            (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH, height=RENDER_HEIGHT,
                                                       viewMatrix=view_matrix, projectionMatrix=proj_matrix,
                                                       renderer=p.ER_BULLET_HARDWARE_OPENGL)
            frame = np.array(px)[:, :, :3]
            return frame
        else:
            return np.array([])

    def getExtendedObservation(self):
        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        goal_pos_1, goal_orn_1 = self._p.getBasePositionAndOrientation(self.goal_object_1.goal)
        goal_pos_2, goal_orn_2 = self._p.getBasePositionAndOrientation(self.goal_object_2.goal)
        goal_pos_3, goal_orn_3 = self._p.getBasePositionAndOrientation(self.goal_object_3.goal)

        self.lidar_readings = self.car.get_lidar_readings()

        if self.observation_callback is not None:
            return self.observation_callback(
                client=self._p,
                car_pos=car_pos,
                car_orn=car_orn,
                goal_pos_1=goal_pos_1,
                goal_orn_1=goal_orn_1,
                goal_pos_2=goal_pos_2,
                goal_orn_2=goal_orn_2,
                goal_pos_3=goal_pos_3,
                goal_orn_3=goal_orn_3,
                lidar_readings=self.lidar_readings
            )
        else:
            raise ValueError("No observation_callback provided!")

    def _termination(self):
        return self._envStepCounter > self.max_steps

    def close(self):
        self._p.disconnect()

    def make_custom_obstacles(self, obstacle_vertices, random_x=0, random_y=0):
        obstacle_vertices_3d = []
        for vertex in obstacle_vertices:
            obstacle_vertices_3d.append((vertex[0], vertex[1], 0.0))
            obstacle_vertices_3d.append((vertex[0], vertex[1], 1.0))

        indices = []
        indices.append([0, 1, 2])
        indices.append([1, 2, 3])
        indices.append([2, 3, 4])
        indices.append([3, 4, 5])
        indices.append([4, 5, 6])
        indices.append([5, 6, 7])
        indices.append([6, 7, 0])
        indices.append([7, 0, 1])
        indices = np.array(indices).astype(np.int32).flatten().tolist()

        col_shape_id = self._p.createCollisionShape(shapeType=self._p.GEOM_MESH, vertices=obstacle_vertices_3d, indices=indices)
        vis_shape_id = self._p.createVisualShape(shapeType=self._p.GEOM_MESH, vertices=obstacle_vertices_3d, indices=indices, rgbaColor=[1, 0, 0, 1])
        obstacle_object = self._p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[-random_x, -random_y, 0]
        )
        self.building_array.append(obstacle_object)

    def collision_detect(self):
        for c in self._p.getContactPoints(bodyA=self.car.car):
            if c[2] != self.plane.get_ids():
                return True
        return False