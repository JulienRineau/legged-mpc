import math

import numpy as np
from numpy import sin, cos

from isaacgym import gymapi, gymutil, gymtorch
from scipy.spatial.transform import Rotation as R
import torch

from stance_controller import StanceController

import util
from params import *

ASSET_ROOT = "../../../assets"
ASSET_FILE = "urdf/a1_description/urdf/a1.urdf"


# # TODO:
# # This is a simplified implementation, and I also feel like is mistaken.
# class Foot:

#     def __init__(self, p_current: np.ndarray, total_time: int, swing_time: int, gait_offset: int):
#         self._current_transform = np.zeros((4,4))

#         self._v_current   = np.zeros(3)
    
#         self._total_time  = total_time
#         self._swing_time  = swing_time
#         self._gait_offset = gait_offset

#         self._vel_com     = np.zeros(3)
#         self._yaw_rate    = 0
        
#         self._robot_feet_pos = ... # This may need to be better specified in another object.
    
#     def get_state_at_time(self, time: int):
#         phase = (time - self._gait_offset) % self._total_time
#         period_count = (time - self._gait_offset + self._swing_time) // self._total_time

#         vel_com = np.array([0.05, 0.00, 0.00])
#         # if foot is in swing phase
#         if phase >= 0 and phase < swing_time:
#             # calculate how far we are in bezier curve

#             origin = self._p_current +     (period_count) * (vel_com) * self._swing_time*(1/60.0)
#             final  = self._p_current + (period_count + 1) * (vel_com) * self._swing_time*(1/60.0)

#             swing_phase = phase / self._swing_time
#             p, v, a = util.computeSwingTrajectoryBezier(swing_phase, self._swing_time*(1/60.0), origin, final)
#             return p, True

#         # if foot is in stance phase
#         else:
#             # prepare for new initialization
#             return self._p_current + (period_count) * (vel_com) * self._swing_time*(1/60.0), False
    
#     def get_future_positions(self, pos: np.ndarray, yaw: float, vel_com: np.ndarray,
#                              yaw_rate: float, time: int, steps: int):
        
#         position_list = np.zeros((3, steps))
#         swing_list = np.zeros(steps)

#         self._vel_com = vel_com
#         self._yaw_rate = yaw_rate

#         self._current_transform = np.block([[R.from_euler('z', yaw), pos],
#                                             [       np.zeros((1,3)),   0]])

#         for i in range(steps):
#             position_list[:, i],  swing_list[i] = self.get_state_at_time(time + i)
#         return position_list, swing_list

class Controller:

    def __init__(self, foot_index):
        # counter
        self.counter = 0
        self.N = 10
        
        # foot information
        self.foot_index = foot_index
        self.swing_start = np.array([0, 60, 60, 0])
        self.swing_time = 60
        self.total_time = 120
        self.offset = np.array([[ 0.172,  0.106,  -0.354],
                                [ 0.172, -0.106,  -0.354],
                                [-0.194,  0.106,  -0.354],
                                [-0.194, -0.106,  -0.354]])

        self.foot_pos_prediction_abs = np.zeros((self.N+1, 4, 3))
        self.foot_vel_prediction_abs = np.zeros((self.N+1, 4, 3))
        self.r_pred = np.zeros((self.N+1, 4, 3))
        self.foot_stance = np.zeros((self.N+1, 4))
        self.foot_prev = np.array([[ 0.172,  0.106,  -0.0],
                                   [ 0.172, -0.106,  -0.0],
                                   [-0.194,  0.106,  -0.0],
                                   [-0.194, -0.106,  -0.0]])
        self.foot_next = np.zeros((4, 3))

        # physical properties
        self.inv_inertia = np.array([[ 5.94025764e+01, -8.78453776e-02, -5.48593283e-01],
                                     [-8.78453851e-02,  1.76745262e+01, -6.05325401e-03],
                                     [-5.48593283e-01, -6.05325028e-03,  1.54577732e+01]])
        self.mass = 4.7129998207092285 # mass

        # positional properties
        self.base_pos = np.zeros(3)
        self.base_rot = np.zeros(3)
        self.base_vel = np.zeros(3)
        self.base_omg = np.zeros(3)

        self.dof_pos = np.zeros((4, 3))
        self.foot_pos_abs = np.zeros((4, 3))

        self.root_pos_pred = np.zeros((self.N+1, 3))
        self.root_vel_pred = np.zeros((self.N+1, 3))
        self.root_ang_pred = np.zeros((self.N+1, 3))
        self.root_rot_pred = np.zeros((self.N+1, 3, 3))
        self.root_w_pred = np.zeros((self.N+1, 3))
        
        # tensor values
        self.jacobian_tensor = jacobian_tensor
        self.root_state_tensor = root_state_tensor
        self.dof_state_tensor = dof_state_tensor
        self.rigid_body_state_tensor = rigid_body_state_tensor

        # controller properties
        self.f_max = 666
        self.f_min = 10
        self.mu = 0.6

        self.Kp = np.diag([20, 20, 20])
        self.Kd = np.diag([10, 10, 10])

        self.force = np.zeros(12)

        L_core = [5,5,1] + [1,1,100] + [0,0,1] + [1,1,0] + [0]
        K_core = [1e-6]*12 # input weight 12 elem.s
        self.stance_controller = StanceController(self.inv_inertia, self.mass,\
                                                  self.mu, self.f_min, self.f_max,\
                                                  DT, self.N,\
                                                  L_core, K_core)
    
    def update_command(self, vel_command=np.zeros(3), rot_command=np.zeros(3)):
        self.desired_height = 0.354
        self.vel_command = vel_command
        self.rot_command = rot_command

    def update_state(self):
        # make sure tensor's are updated
        # predict foot positions
        # build matricies
        self.root_pos = self.root_state_tensor[0, 0:3].numpy()
        self.root_quat = self.root_state_tensor[0, 3:7].numpy()
        self.root_lin_vel = self.root_state_tensor[0, 7:10].numpy()
        self.root_ang_vel = self.root_state_tensor[0, 10:13].numpy()
        self.joint_pos = self.dof_state_tensor[:, 0].numpy()
        self.joint_vel = self.dof_state_tensor[:, 1].numpy()

        r = R.from_quat(self.root_quat)
        self.root_euler = r.as_euler('xyz')
        self.rot_mat = r.as_matrix()
        r = R.from_euler('z', self.root_euler[2])
        self.rot_mat_z = r.as_matrix()

        self.root_lin_vel_rel = self.root_lin_vel @ self.rot_mat_z
        self.root_ang_vel_rel = self.root_ang_vel @ self.rot_mat

        foot_radius = 0.0 # 0.0265
        self.foot_pos_world = self.rigid_body_state_tensor[foot_index, 0:3].numpy()
        self.foot_pos_world[:, 2] -= foot_radius
        self.foot_pos_abs = self.foot_pos_world - self.root_pos
        self.foot_pos_rel = self.foot_pos_abs @ self.rot_mat
        self.foot_vel_world = self.rigid_body_state_tensor[self.foot_index, 3:6].numpy()
        self.foot_vel_abs = self.foot_vel_world - self.root_lin_vel  # TODO: miss the angular velocity
        self.foot_vel_rel = self.foot_vel_abs @ self.rot_mat
        foot_jaco = self.jacobian_tensor[0, foot_index, 0:3, 6:].numpy()
        foot_jaco = np.repeat(self.rot_mat.T[None, ...], 4, axis=0) @ foot_jaco
        self.foot_jaco = foot_jaco.reshape(12, 12)

        # build the predicted states of the body based off the velocity command
        self._build_root_pred()

        # update foot states (stance or swing)
        self._update_foot_states()
        self._build_foot_pred()

        for i in range(4):
            self.r_pred[:, i, :] = self.foot_pos_prediction_abs[:, i, :] - self.root_pos_pred[:, :]

        print(self.foot_stance)

        # print(self.root_pos_pred)

    
    def request_torques(self):
        # get ground reaction forces
        # ground reaction forces get turned into torque values

        R_yaw_aligned = self.rot_mat_z.T @ self.rot_mat
        # FIXME:warren: in update_x0, position should be [0,0,z], but it turned out that [0,0,0] works perfectly while [0,0,z] will cause robot to squat down, no idea why
        self.stance_controller.update_x0(self.root_euler,
                                         self.root_pos,
                                         self.root_ang_vel,
                                         self.root_lin_vel)
        self.stance_controller.update_ref(self.root_ang_pred,
                                          self.root_pos_pred,
                                          self.root_rot_pred,
                                          self.root_w_pred,
                                          self.root_vel_pred,
                                          self.r_pred,
                                          self.foot_stance)
        self.force = self.stance_controller.run_MPC() @ self.rot_mat_z.T # rot_mat_z = W_R_yawAlignedW, this line transforms grf to global world frame
        self.force = self.force.reshape((12,))

        for i in range(4):
            if not self.foot_stance[0, i]: # if foot is in swing we need to implement PD controller
                p_desired = self.foot_pos_prediction_abs[0, i, :]
                v_desired = self.foot_vel_prediction_abs[0, i, :]

                p_actual = self.foot_pos_world[i, :]
                v_actual = self.foot_vel_world[i, :]

                foot_force = -self.Kp@(p_actual - p_desired) - self.Kd@(v_actual - v_desired)

                self.force[3*i:3*(i+1)] = foot_force
        
        self.counter += 1

        torques = self.foot_jaco.T @ self.force

        return torques

    def _build_root_pred(self):
        for k in range(self.N+1):
            x, y, yaw = self._pred_base_pos(k*DT)
            self.root_pos_pred[k, :] = np.array([x, y, self.desired_height])
            self.root_ang_pred[k, :] = np.array([0, 0, yaw])
            r = R.from_euler('z', self.root_euler[2]).as_matrix()
            self.root_rot_pred[k, :, :] = r
            self.root_vel_pred[k, :] = r@self.vel_command
            self.root_w_pred[k, :] = self.rot_command

    def _pred_base_pos(self, t):
        x0 = 0 # self.root_pos[0]
        y0 = 0 # self.root_pos[1]
        yaw0 = 0 # self.root_euler[2]
        vel_rel = self.rot_mat_z @ self.vel_command
        vel_x = vel_rel[0]
        vel_y = vel_rel[1]
        w = self.rot_command[2]

        if w < 1e-3:
            x = x0 + vel_x*t
            y = y0 + vel_y*t
            yaw = y0
        else:
            x = x0 + (vel_x/w)*(np.sin(w*t + yaw0) - np.sin(yaw0)) + (vel_y/w)*(np.cos(w*t + yaw0) - np.cos(w*t + yaw0))
            y = y0 + (vel_x/w)*(-np.cos(w*t + yaw0) + np.cos(yaw0)) + (vel_y/w)*(np.sin(w*t + yaw0) - np.sin(w*t + yaw0))
            yaw = yaw0 + w*t
        return x, y, yaw

    def _update_foot_states(self):
        for i in range(4):
            t_curr = (self.counter - self.swing_start[i]) % self.total_time
            if t_curr < self.swing_time:
                self.foot_stance[0, i] = False
                t_next = (self.swing_time - t_curr)
            else:
                self.foot_stance[0, i] = True
                t_next = (self.swing_time + self.total_time - t_curr)
                self.foot_prev[i] = self.foot_pos_world[i, :] # self.offset[i] + np.array([0, 0, 0.354]) 
                print(self.foot_prev[i], i)

            x, y, yaw = self._pred_base_pos(t_next*DT)
            foot_next = np.array([x + self.offset[i, 0] * np.cos(yaw) - self.offset[i, 1] * np.sin(yaw),
                                  y + self.offset[i, 0] * np.sin(yaw) + self.offset[i, 1] * np.cos(yaw),
                                  self.desired_height + self.offset[i, 2]])
            self.foot_next[i] = foot_next

            print(self.foot_prev[i], i)
            print(self.foot_next[i], i)
    
    def _build_foot_pred(self):
        for k in range(self.N + 1):
            for i in range(4):
                t_curr = (self.counter + k - self.swing_start[i]) % self.total_time

                if self.foot_stance[0, i]:
                    if t_curr < self.swing_time: # future position is in swing
                        (p, v) = util.computeSwingTrajectoryBezier(self.foot_prev[i], self.foot_next[i], t_curr / self.swing_time, self.swing_time*DT, self.desired_height*0.5)
                        self.foot_pos_prediction_abs[k, i, :] = p
                        self.foot_vel_prediction_abs[k, i, :] = v
                        self.foot_stance[k, i] = False
                    else:
                        self.foot_pos_prediction_abs[k, i, :] = self.foot_prev[i]
                        self.foot_vel_prediction_abs[k, i, :] = np.zeros(3)
                        self.foot_stance[k, i] = True
                else:
                    if t_curr < self.swing_time: # future position is in swing
                        (p, v) = util.computeSwingTrajectoryBezier(self.foot_prev[i], self.foot_next[i], t_curr / self.swing_time, self.swing_time*DT, self.desired_height*0.5)
                        self.foot_pos_prediction_abs[k, i, :] = p
                        self.foot_vel_prediction_abs[k, i, :] = v
                        self.foot_stance[k, i] = False
                    else:
                        self.foot_pos_prediction_abs[k, i, :] = self.foot_next[i]
                        self.foot_vel_prediction_abs[k, i, :] = np.zeros(3)
                        self.foot_stance[k, i] = True

    def _pred_foot_pos(self):
        return ...
    
    def _build_step_matricies(self):
        return ...
    
    def _form_QP(self):
        return ...
    
    def _solve(self):
        return ...

# Parse Arguments
args = gymutil.parse_arguments(
    description="Joint monkey: Animate degree-of-freedom ranges",
    custom_parameters=[
        {"name": "--speed_scale", "type": float, "default": 1.0, "help": "Animation speed scale"},
        {"name": "--show_axis", "action": "store_true", "help": "Visualize DOF axis"}])

# Setup Isaac Gym Environment

# Initialize
gym = gymapi.acquire_gym()

# Simulation Parameters
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0

if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

sim_params.use_gpu_pipeline = False

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# Add Ground Plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
plane_params.static_friction = 1.0
plane_params.dynamic_friction = 1.0
gym.add_ground(sim, plane_params)

# Create Viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

# Load Assets
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = False
asset_options.flip_visual_attachments = True
asset_options.use_mesh_materials = True
asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT

asset = gym.load_asset(sim, ASSET_ROOT, ASSET_FILE, asset_options)

# Store Info of Robot Joints
dof_names = gym.get_asset_dof_names(asset)
dof_props = gym.get_asset_dof_properties(asset)

dof_props['stiffness'] = np.zeros((12,))

num_dofs = gym.get_asset_dof_count(asset)
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
dof_types = [gym.get_asset_dof_type(asset, i) for i in range(num_dofs)]

# DOF Positions and Velocities
dof_positions = dof_states['pos']
dof_velocities = dof_states['vel']

# Setup Camera
cam_pos = gymapi.Vec3(1.0, 2.0, 1.0)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# Setup Environment
spacing = 5
env_lower = gymapi.Vec3(-spacing, 0.0, -spacing)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
env = gym.create_env(sim, env_lower, env_upper, 1)

# Starting Pose
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.4)
# pose.r = gymapi.Quat(-0.707107, 0.0, 0.0, 0.707107)

# Create Robot "Actor"
actor_handle = gym.create_actor(env, asset, pose, 'actor')
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)


gym.set_actor_dof_properties(env, actor_handle, dof_props)

# Acquire Properties of Robot
link_dict = gym.get_asset_rigid_body_dict(asset)
dof_dict = gym.get_asset_dof_dict(asset)
joint_dict = gym.get_asset_joint_dict(asset)


rigid_body_prop = gym.get_actor_rigid_body_properties(env, actor_handle) # base should be the first rigid-body listed
base_prop = rigid_body_prop[link_dict['trunk']]

base_inv_inertia = base_prop.invInertia
base_inertia = base_prop.inertia
base_mass = base_prop.mass

base_inertia = np.array([[base_inertia.x.x, base_inertia.x.y, base_inertia.x.z],
                         [base_inertia.y.x, base_inertia.y.y, base_inertia.y.z],
                         [base_inertia.z.x, base_inertia.z.y, base_inertia.z.z]])

base_inv_inertia = np.array([[base_inv_inertia.x.x, base_inv_inertia.x.y, base_inv_inertia.x.z],
                             [base_inv_inertia.y.x, base_inv_inertia.y.y, base_inv_inertia.y.z],
                             [base_inv_inertia.z.x, base_inv_inertia.z.y, base_inv_inertia.z.z]])

print(base_inv_inertia)

print(base_mass)

foot_names = ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
foot_index = [link_dict[name] for name in foot_names]


hip_names = ['FL_hip', 'FR_hip', 'RL_hip', 'RR_hip']
hip_index = [link_dict[name] for name in hip_names]

# Define Tensors and Information we are going to grab from simulation.
_jacobian_tensor = gym.acquire_jacobian_tensor(sim, 'actor')
jacobian_tensor = gymtorch.wrap_tensor(_jacobian_tensor)

_dof_state_tensor = gym.acquire_dof_state_tensor(sim)
dof_state_tensor = gymtorch.wrap_tensor(_dof_state_tensor)

_root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
root_state_tensor = gymtorch.wrap_tensor(_root_state_tensor)

_rigid_body_state_tensor = gym.acquire_rigid_body_state_tensor(sim)
rigid_body_state_tensor = gymtorch.wrap_tensor(_rigid_body_state_tensor)

t_step = 0

# # [0.183, 0.047, 0]

# x, y, z = (0.05, 0.15, -0.35)

# pDes1 = np.array([[x,  y, z],
#                   [x, -y, z],
#                   [-x,  y, z],
#                   [-x, -y, z]])

# pDes2 = np.array([[0.283, 0.047, -0.2],
#                   [0.283, -0.047, -0.2],
#                   [0.283, 0.047, -0.2],
#                   [0.283, -0.047, -0.2]])

# pCurr = pDes1
# pFinal = pDes2

contact_offsets = [0, 100, 110, 10]
total_time = 120
swing_time = 60

# foot_list = [Foot(pDes1[i], total_time, swing_time, contact_offsets[i]) for i in range(4)]

a1_controller = Controller(foot_index)

# Run
while not gym.query_viewer_has_closed(viewer):
    # Simulate in Isaac Gym
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Calculate Torques
    torques = np.zeros((12,), dtype='float32')

    gym.refresh_jacobian_tensors(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)

    a1_controller.update_command(vel_command = np.array([0, 0, 0]))
    a1_controller.update_state()

    torques = a1_controller.request_torques()

    # print('Final Torque: {}'.format(torques))


    print(torques)

    list = [0]*12

    for i in range(12):
        list[i] = torques[i]

    # Apply to Simulation
    gym.apply_actor_dof_efforts(env, actor_handle, list)

    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)

    t_step += 1

    gym.sync_frame_time(sim)

print("Finished ... Exiting")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
