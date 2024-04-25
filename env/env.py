'''
Author: Zihao Wang wzh7076@gmail.com
Date: 2024-04-15 19:03:44
LastEditors: Zihao Wang wzh7076@gmail.com
LastEditTime: 2024-04-19 16:13:07
FilePath: \Marine-Vehicle-Simulation-Environments-For-Deep-Reinforcement-Learning\env\env.py
Description: 

'''

import gym
from abc import ABC
from vehicles import *
from lib import *

class Vehicle_env(gym.Env, ABC):
    def __init__(self, vehicle_name='remus100', plot=False, ):
        self.vehicle_name = vehicle_name
        self.plot = plot
        self.sampleTime = 0.1
        self.sampleFreq = 1/self.sampleTime
        self.maxStep = 1000
        self.DOF = 6
        self.step_count = 0
        
        # vehicle state space
        self.eta = None  # position/attitude, user editable
        self.nu = None  # velocity, defined by vehicle class
        self.u_actual = None   # actual inputs, defined by vehicle class
        self.state = None
        self.dimU = None
        
        
        # history info
        self.rewards_array = []
        self.total_steps = 0
        
        # vehicle class
        self.vehicle = self.select_vehicle(vehicle_name)
        # aciton space is the same as the u_actual
        self.controls = self.vehicle.controls
        self.controls_range = self.vehicle.controls_range
        #  action space
        self.min_action = np.array([self.controls_range[control][0] for control in self.controls])
        self.max_action = np.array([self.controls_range[control][1] for control in self.controls])
        
        self.action_space = gym.spaces.Box(
            low=self.min_action, high=self.max_action, shape=(2,), dtype=np.float32
        )
        
        
    def reset(self):
        raise NotImplementedError
    
    def step(self, action):
        raise NotImplementedError
    
    def render(self):
        raise NotImplementedError
    
    def select_vehicle(self, vehicle_name):
        type2vehicle = {
            'DSRV': DSRV(),
            'frigate': frigate(),
            'otter': otter(),
            'ROVzefakkel': ROVzefakkel(),
            'semisub': semisub(),
            'shipClarke83': shipClarke83(),
            'supply': supply(),
            'tanker': tanker(),
            'remus100': remus100()
        }
        return type2vehicle[vehicle_name]

    def attitudeEuler(self):
        """
        eta = attitudeEuler(eta,nu,sampleTime) computes the generalized 
        position/Euler angles eta[k+1]
        """
        return attitudeEuler(self.eta, self.nu, self.sampleTime)


if __name__ == '__main__':
    create_env = Vehicle_env()
    print(create_env.vehicle_name)
