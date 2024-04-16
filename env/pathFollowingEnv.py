'''
Author: Zihao Wang wzh7076@gmail.com
Date: 2024-04-15 20:16:10
LastEditors: Zihao Wang wzh7076@gmail.com
LastEditTime: 2024-04-16 16:19:32
FilePath: \Vehicle_Drl\env\pathFollowingEnv.py
Description: 

'''


from env import Vehicle_env
import numpy as np
from gym import spaces
from lib.path import Path, PolylinePath, SplinePath

type2Path = {
    'polyline': PolylinePath,
    'spline': SplinePath,
}

class PathFollowingEnv(Vehicle_env):
    def __init__(self, vehicle_name='DSRV', plot=False,  path_type='polyline',  **kwargs):
        super(PathFollowingEnv, self).__init__(vehicle_name, plot)
        super(PathFollowingEnv, self).reset()
        
        self.path_type = path_type
        self.path = None
        self.cross_track_error = 0
        
        self.low_state = np.array([0, 0, 0, -np.pi, -np.pi, -np.pi], dtype=np.float32)
        self.high_state = np.array([np.inf, np.inf, np.inf, np.pi, np.pi, np.pi], dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        
        # aciton space is the same as the u_actual
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32
        )
        self.action_space = self.vehicle.action_space
    
    def reset(self, path_args=None):
        # init the state of the vehicle
        super(PathFollowingEnv, self).reset()
        [init_x, init_y, init_z] = [0,0,0]
        [init_phi, init_theta, init_psi] = [0,0,0]
        self.eta = np.array([init_x, init_y, init_z, init_phi, init_theta, init_psi], float)
        
        
        # 初始化path
        if self.path_type == 'polyline':
            self.path = PolylinePath()
        elif self.path_type == 'spline':
            self.path = SplinePath()
        # Add additional path types as needed
        else:
            raise ValueError(f"Unrecognized path type: {self.path_type}")
        
        # init the path following error
        
        
        return self.eta
   
    def step(self, action):
        # the action should in the same space as the u_actual
        # update the state by vehicle.dynamic()
        # the vehicle.dynamic() should be defined in the vehicle class
        # [nu, u_actual] = dynamics(eta,nu,u_actual,u_control,sampleTime)
        [nu, u_actual] = self.vehicle.dynamics(self.eta, self.nu, self.u_actual, action, self.sampleTime)
        self.nu = nu
        self.u_actual = u_actual
        self.eta = self.attitudeEuler()
        reward = self.compute_reward()
        done = self.judge_done()
        
        return self.nu, reward, done, {}
    
    def compute_error(self):
        return 0
    
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
        
        print(env.step(u_control))
        
        
    