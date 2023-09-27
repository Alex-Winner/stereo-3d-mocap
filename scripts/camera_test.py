import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R

mesh_ = o3d.geometry.TriangleMesh()
mesh_.create_sphere

geom1 = mesh_.create_sphere()
geom1.translate((-10, 0, 0))
geom2 = mesh_.create_sphere()
geom2.translate((10, 0, 0))

vis = o3d.visualization.Visualizer()
vis.create_window()

vis.add_geometry(geom1)
vis.add_geometry(geom2)

ctrl = vis.get_view_control()

while vis.poll_events():
    # rotate the camera
    camera_params = ctrl.convert_to_pinhole_camera_parameters()
    rot = np.eye(4)
    rot[:3, :3] = R.from_euler('y', 5, degrees=True).as_matrix()
    rot = rot.dot(camera_params.extrinsic)
    camera_params.extrinsic = rot
    ctrl.convert_from_pinhole_camera_parameters(camera_params)

    vis.update_renderer()

vis.destroy_window()