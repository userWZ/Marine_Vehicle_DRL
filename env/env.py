'''
Author: 王子豪 as456741@qq.com
Date: 2024-04-15 19:03:44
LastEditors: 王子豪 as456741@qq.com
LastEditTime: 2024-04-16 16:02:46
FilePath: \Vehicle_Drl\env\env.py
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
        
        # vehicle state space
        self.eta = None  # position/attitude, user editable
        self.nu = None  # velocity, defined by vehicle class
        self.u_actual = None   # actual inputs, defined by vehicle class
        self.state = None
        self.dimU = None
        
        # aciton space is the same as the u_actual
        
        # controls/actions array
        self.controls_array = []
        
        # history info
        self.rewards_array = []
        self.total_steps = 0
        
        # vehicle class
        self.vehicle = self.select_vehicle(vehicle_name)
        self.reset()
        
    def reset(self):
        self.dimU = self.vehicle.dimU
        self.eta = np.zeros(self.DOF,float)
        self.nu =  self.vehicle.nu # velocity vector
        self.u_actual = self.vehicle.u_actual 
        return self.state
    
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
