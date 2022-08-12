"""Examples with 3D mesh rendering using various backend and comparison with deodr."""

import os

import deodr
from deodr import differentiable_renderer
from deodr.triangulated_mesh import ColoredTriMesh

import imageio

import matplotlib.pyplot as plt

import numpy as np

from scipy.spatial.transform import Rotation

import trimesh


obj_filename = "filled_not_boerner.obj"


def default_scene(
    obj_file, width=640, height=480, use_distortion=True, integer_pixel_centers=True
):

    mesh_trimesh = trimesh.load(obj_file)

    mesh = ColoredTriMesh.from_trimesh(mesh_trimesh)

    # rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()
    rot = Rotation.from_euler("xyz", [180, 0, 0], degrees=True).as_matrix()

    camera = differentiable_renderer.default_camera(
        width, height, 80, mesh.vertices, rot
    )
    if use_distortion:
        camera.distortion = np.array([-0.5, 0.5, 0, 0, 0])

    bg_color = np.array((0.8, 0.8, 0.8))
    scene = differentiable_renderer.Scene3D(integer_pixel_centers=integer_pixel_centers)
    light_ambient = 0
    light_directional = 0.4 * np.array([1, -3, -0.5])
    scene.set_light(light_directional=light_directional, light_ambient=light_ambient)
    scene.set_mesh(mesh)
    scene.set_background_color(bg_color)
    return scene, camera


def example_rgb(display=True, save_image=False, width=640, height=480):
    obj_file = obj_filename
    scene, camera = default_scene(obj_file, width=width, height=height)
    scene.mesh.texture = np.array([
        [[1, 0, 0], [1, 0, 0]],
        [[1, 0, 1], [0, 0, 1]]
    ])
    image = scene.render(camera)
    if save_image:
        image_file = "filled_not_boerner.png"
        os.makedirs(os.path.dirname(image_file), exist_ok=True)
        image_uint8 = (image * 255).astype(np.uint8)
        imageio.imwrite(image_file, image_uint8)
    if display:
        plt.figure()
        plt.title("deodr rendering")
        plt.imshow(image)
    return image


def example_channels(display=True, save_image=False, width=640, height=480):
    obj_file = obj_filename
    scene, camera = default_scene(obj_file, width=width, height=height)

    def normalize(v):
        if v.ndim == 3 and v.shape[2] < 3:
            nv = np.zeros((v.shape[0], v.shape[1], 3))
            nv[:, :, : v.shape[2]] = v
        else:
            nv = v
        return (nv - nv.min()) / (nv.max() - nv.min())

    scene.sigma = 0

    channels = scene.render_deferred(camera)
    if display:
        plt.figure()
        for i, (name, v) in enumerate(channels.items()):
            ax = plt.subplot(2, 4, i + 1)
            ax.set_title(name)
            ax.imshow(normalize(v))

    if save_image:
        for name, v in channels.items():
            image_file = os.path.abspath(
                os.path.join(deodr.data_path, f"test/duck_{name}.png")
            )
            os.makedirs(os.path.dirname(image_file), exist_ok=True)
            image_uint8 = (normalize(v) * 255).astype(np.uint8)
            imageio.imwrite(image_file, image_uint8)


if __name__ == "__main__":
    #example_moderngl(display=True)
    example_rgb(save_image=False)
    example_channels(save_image=False)
    #example_pyrender()
    plt.show()