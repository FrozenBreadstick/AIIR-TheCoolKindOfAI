import pybullet as p
import os


class Plane:
    def __init__(self, client):
        f_name = os.path.join(os.path.dirname(__file__), 'simpleplane.urdf')
        self.plane_id = client.loadURDF(fileName=f_name,
                                   basePosition=[0, 0, 0])


    def get_ids(self):
        return self.plane_id

