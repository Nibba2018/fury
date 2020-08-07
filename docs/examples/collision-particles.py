"""
================================
Collisions of particles in a box
================================

<<<<<<< HEAD
This is a simple demonstration of how you can simulate moving
particles in a box using FURY.
"""

##############################################################################
# In this example, the particles collide with each other and with the walls
# of the container. When the the collision happens between two particles,
# the particle with less velocity changes its color and gets the same color
# as the particle with higher velocity. For simplicity, in this demo we
# do not apply forces.

import numpy as np
from fury import window, actor, ui, utils
import itertools


##############################################################################
# Here, we define the edges of the box.


def box_edges(box_lx, box_ly, box_lz):

    edge1 = 0.5 * np.array([[box_lx, box_ly, box_lz],
                            [box_lx, box_ly, -box_lz],
                            [-box_lx, box_ly, -box_lz],
                            [-box_lx, box_ly, box_lz],
                            [box_lx, box_ly, box_lz]])
    edge2 = 0.5 * np.array([[box_lx, box_ly, box_lz],
                            [box_lx, -box_ly, box_lz]])
    edge3 = 0.5 * np.array([[box_lx, box_ly, -box_lz],
                            [box_lx, -box_ly, -box_lz]])
    edge4 = 0.5 * np.array([[-box_lx, box_ly, -box_lz],
                            [-box_lx, -box_ly, -box_lz]])
    edge5 = 0.5 * np.array([[-box_lx, box_ly, box_lz],
                            [-box_lx, -box_ly, box_lz]])
    lines = [edge1, -edge1, edge2, edge3, edge4, edge5]
    return lines


##############################################################################
# Here we define collision between walls-particles and particle-particle.
# When collision happens, the particle with lower velocity gets the
# color of the particle with higher velocity

def collision():
    global vel, xyz, num_particles
    num_vertices = vertices.shape[0]
    sec = np.int(num_vertices / num_particles)

    for i, j in np.ndindex(num_particles, num_particles):

        if (i == j):
            continue
        distance = np.linalg.norm(xyz[i] - xyz[j])
        vel_mag_i = np.linalg.norm(vel[i])
        vel_mag_j = np.linalg.norm(vel[j])
        # Collision happens if the distance between the centars of two
        # particles is less or equal to the sum of their radii
        if (distance <= (radii[i] + radii[j])):
            vel[i] = -vel[i]
            vel[j] = -vel[j]
            if vel_mag_j > vel_mag_i:
                vcolors[i * sec: i * sec + sec] = \
                    vcolors[j * sec: j * sec + sec]
            if vel_mag_i > vel_mag_j:
                vcolors[j * sec: j * sec + sec] = \
                    vcolors[i * sec: i * sec + sec]
            xyz[i] = xyz[i] + vel[i] * dt
            xyz[j] = xyz[j] + vel[j] * dt
    # Collision between particles-walls;
    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + radii[:]) |
                          (xyz[:, 0] >= (0.5 * box_lx - radii[:]))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + radii[:]) |
                          (xyz[:, 1] >= (0.5 * box_ly - radii[:]))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + radii[:]) |
                          (xyz[:, 2] >= (0.5 * box_lz - radii[:]))),
                         - vel[:, 2], vel[:, 2])


##############################################################################
# We define position, velocity, color and radius randomly for 50 particles
# inside the box.

global xyz, dt, steps, num_particles, vel, vertices
num_particles = 50
box_lx = 20
box_ly = 20
box_lz = 10
steps = 1000
dt = 0.05
xyz = np.array([box_lx, box_ly, box_lz]) * (np.random.rand(num_particles, 3)
                                            - 0.5) * 0.6
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.random.rand(num_particles, 3)
radii = np.random.rand(num_particles) + 0.01

##############################################################################
# With box, streamtube and sphere actors, we can create the box, the
# edges of the box and the spheres respectively.

scene = window.Scene()
=======
"""

import numpy as np
import sys
import math
from fury import window, actor, ui, utils, pick
import itertools
from vtk.util import numpy_support


def boundary_conditions():
    global vel, vcolors, xyz, num_particles
    num_vertices = vertices.shape[0]
    color_add = np.array([10, 0, 0], dtype='uint8')
    no_vertices_per_sphere = len(vertices)/num_particles
    sec = np.int(num_vertices / num_particles)
    for i,j in np.ndindex(num_particles, num_particles):
        if (i == j):
            continue
        distance = (((xyz[i, 0]-xyz[j, 0])**2) + ((xyz[i, 1]-xyz[j, 1])**2) + ((xyz[i, 1]-xyz[j, 1])**2))** 0.5
        if (distance <= (radii[i] + radii[j])):
            vel[i] = -vel[i]
            vel[j] = -vel[j]
            vcolors[i * sec: i * sec + sec] += color_add
         #   vcolors[j * sec: j * sec + sec] += color_add
            xyz[i] = xyz[i] + vel[i] * dt
            xyz[j] = xyz[j] + vel[j] * dt




    vel[:, 0] = np.where(((xyz[:, 0] <= - 0.5 * box_lx + radii[:]) |
                          (xyz[:, 0] >= (0.5 * box_lx - radii[:]))),
                         - vel[:, 0], vel[:, 0])
    vel[:, 1] = np.where(((xyz[:, 1] <= - 0.5 * box_ly + radii[:]) | (xyz[:, 1] >= (0.5 * box_ly - radii[:]))),
                         - vel[:, 1], vel[:, 1])
    vel[:, 2] = np.where(((xyz[:, 2] <= -0.5 * box_lz + radii[:]) | (xyz[:, 2] >= (0.5 * box_lz - radii[:]))),
                         - vel[:, 2], vel[:, 2])


global xyz, dt, steps, num_particles, vel, vcolors

num_particles = 200
box_lx = 50
box_ly = 50
box_lz = 50
steps = 1000
dt = 0.5

xyz = (box_lz * 0.75) * (np.random.rand(num_particles, 3) - 0.5)
vel = 4 * (np.random.rand(num_particles, 3) - 0.5)
colors = np.zeros((num_particles, 3)) + np.array([1, 1, 1])
radii = np.random.rand(num_particles) + 0.01

scene = window.Scene()

>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c
box_centers = np.array([[0, 0, 0]])
box_directions = np.array([[0, 1, 0]])
box_colors = np.array([[255, 255, 255]])
box_actor = actor.box(box_centers, box_directions, box_colors,
                      scale=(box_lx, box_ly, box_lz))
<<<<<<< HEAD
utils.opacity(box_actor, 0.2)
scene.add(box_actor)

lines = box_edges(box_lx, box_ly, box_lz)
line_actor = actor.streamtube(lines, colors=(1, 0.5, 0), linewidth=0.1)
scene.add(line_actor)
=======

utils.opacity(box_actor, 0.5)
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c

sphere_actor = actor.sphere(centers=xyz,
                            colors=colors,
                            radii=radii)
scene.add(sphere_actor)
<<<<<<< HEAD

showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=True,
                           order_transparent=True)
showm.initialize()
tb = ui.TextBlock2D(bold=True)
scene.zoom(0.8)
=======
#scene.add(actor.axes(scale=(0.5*box_lx, 0.5*box_ly, 0.5*box_lz)))
showm = window.ShowManager(scene,
                           size=(900, 768), reset_camera=False,
                           order_transparent=True)
showm.initialize()
scene.add(box_actor)
tb = ui.TextBlock2D(bold=True)
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c

# use itertools to avoid global variables
counter = itertools.count()

<<<<<<< HEAD
=======
global vertices, vcolors
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c
vertices = utils.vertices_from_actor(sphere_actor)
vcolors = utils.colors_from_actor(sphere_actor, 'colors')
no_vertices_per_sphere = len(vertices)/num_particles
initial_vertices = vertices.copy() - \
    np.repeat(xyz, no_vertices_per_sphere, axis=0)

<<<<<<< HEAD

=======
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c
def timer_callback(_obj, _event):
    global xyz, dt, steps, num_particles, vcolors, vel
    cnt = next(counter)
    tb.message = "Let's count up to 1000 and exit :" + str(cnt)
<<<<<<< HEAD
    xyz = xyz + vel * dt
    collision()

    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    utils.update_actor(sphere_actor)

    scene.reset_clipping_range()
    # scene.azimuth(0.1)
=======
    boundary_conditions()
    xyz = xyz + vel * dt

    vertices[:] = initial_vertices + \
        np.repeat(xyz, no_vertices_per_sphere, axis=0)
    # Tell actor that memory is modified
    utils.update_actor(sphere_actor)
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c
    showm.render()

    if cnt == steps:
        showm.exit()


scene.add(tb)
<<<<<<< HEAD
showm.add_timer_callback(True, 50, timer_callback)
=======
showm.add_timer_callback(True, 100, timer_callback)
>>>>>>> 22224ff1c701bd8796146c0ac72129a511c77e8c
showm.start()
