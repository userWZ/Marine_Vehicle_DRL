'''
Author: Zihao Wang wzh7076@gmail.com
Date: 2024-04-15 20:16:10
LastEditors: Zihao Wang wzh7076@gmail.com
LastEditTime: 2024-04-19 17:01:52
FilePath: \Marine-Vehicle-Simulation-Environments-For-Deep-Reinforcement-Learning\env\pathFollowingEnv.py
Description: 

'''
from env import Vehicle_env
import numpy as np
from gym import spaces
from lib.path import Path, PolylinePath, SplinePath, SineWavePath3D
from lib.plotTimeSeries import *
type2Path = {
    'polyline': PolylinePath,
    'spline': SplinePath,
    'sine_wave': SineWavePath3D
}

class PathFollowingEnv(Vehicle_env):
    def __init__(self, vehicle_name='DSRV', plot=False,  path_type='polyline'):
        super(PathFollowingEnv, self).__init__(vehicle_name, plot)
        
        self.path_type = path_type
        self.path = None
        self.cross_track_error = []
        
        self.low_state = np.array([0, 0, 0, -np.pi, -np.pi, -np.pi], dtype=np.float32)
        self.high_state = np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        
        # 动作维度
        self.action_dim = self.vehicle.dimU
        
        # aciton space is the same as the u_actual
        # self.action_space = spaces.Box(
        #     low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        # )
        # self.action_space = self.vehicle.action_space
        
        self.history = None
        self.reset()
    
    def reset(self, path_args=None):
        # init the state of the vehicle
        self.dimU = self.vehicle.dimU
        self.eta = np.zeros(self.DOF,float)
        self.nu =  self.vehicle.nu # velocity vector
        self.u_actual = self.vehicle.u_actual # actual inputs
        [init_x, init_y, init_z] = [0,0,0]
        [init_phi, init_theta, init_psi] = [0,0,0]
        self.eta = np.array([init_x, init_y, init_z, init_phi, init_theta, init_psi], float)
        
        self.state = np.concatenate((self.eta, self.nu, self.u_actual))
        self.history = self.state
        # 初始化path
        if self.path_type == 'polyline':
            points = [[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 3, 3]]
            self.path = PolylinePath(points)
        elif self.path_type == 'spline':
            self.path = SplinePath()
        elif self.path_type == 'sine_wave':
            self.path = SineWavePath3D()
        else:
            raise ValueError(f"Unrecognized path type: {self.path_type}")
        
        return self.state
   
    def step(self, action):
        # the action should in the same space as the u_actual
        # update the state by vehicle.dynamic()
        # the vehicle.dynamic() should be defined in the vehicle class
        # [nu, u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime)
        self.step_count += 1
        [nu, u_actual] = self.vehicle.dynamics(self.eta, self.nu, self.u_actual, action, self.sampleTime)
        self.nu = nu
        self.u_actual = u_actual
        self.eta = self.attitudeEuler()
        
        self.position = self.eta[0:3]   
        self.velocity = self.nu[0:3]
        
        # 计算横向偏差
        closest_point, distance = self.path.get_closest_point(self.position)
        
        self.cross_track_error.append(distance)
        
        reward = self.compute_reward()
        
        done = self.judge_done()
        
        self.state = np.concatenate((self.eta, self.nu, self.u_actual))
        self.history = np.vstack([self.history, self.state])
        
        return self.state, reward, done, {}

    def compute_reward(self):
        return 0
    
    def judge_done(self):
        return False
    
    def render(self):
        pass
    
    

if __name__ == '__main__':
    import math
    env = PathFollowingEnv()
    env.reset()
    delta_c = 20 * (math.pi / 180)
    for t in range(100):
        if t > 30:
            delta_c = 10 * (math.pi / 180)
        if t > 50:
            delta_c = 0
        u_control = np.array([delta_c], float)
        
        state, reward, done, _ = env.step(u_control)
    
    simTime = np.arange(start=0, stop=env.step_count*env.sampleTime+env.sampleTime, step=env.sampleTime)[:, None]
    plotVehicleStates(simTime, env.history, 1)
    numDataPoints = 50
    plot3D(env.history, 50, 10, '3D_animation.gif', 3)  
    plt.show()
        
    