import pybullet as p

class Building:
    def __init__(self, client, ):
        self.client = client
        # Create a red cylinder (Radius 0.5, Height 1.0)
        col_shape_id = client.createCollisionShape(shapeType=client.GEOM_MESH)
        vis_shape_id = client.createVisualShape(shapeType=client.GEOM_MESH, rgbaColor=[1, 0, 0, 1]) # Red
        
        self.obstacle = client.createMultiBody(
            baseMass=0, # Infinite mass, completely static
            baseCollisionShapeIndex=col_shape_id,
            baseVisualShapeIndex=vis_shape_id,
            basePosition=[base_position[0], base_position[1], 0.5]
        )
