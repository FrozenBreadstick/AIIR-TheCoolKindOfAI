from http import client

import pybullet as p
import os


class Goal:
    def __init__(self, client, base):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplegoal.urdf')
        self.goal = client.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0])
        # Make goal invisible to LiDAR rays (and optionally collisions)
        # print(p.setCollisionFilterGroupMask.__doc__)
        self.client.setCollisionFilterGroupMask(
            self.goal,
            -1,
            0,
            0
        )
        num_links = self.client.getNumJoints(self.goal)

        for link_index in range(-1, num_links):
            self.client.setCollisionFilterGroupMask(
                self.goal,
                link_index,
                collisionFilterGroup=0,   # this object belongs to no group
                collisionFilterMask=0     # this object collides with nothing
            )

        self.position = [base[0], base[1], 0]



