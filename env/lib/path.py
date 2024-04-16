'''
Author: 王子豪 as456741@qq.com
Date: 2024-04-16 13:41:07
LastEditors: 王子豪 as456741@qq.com
LastEditTime: 2024-04-16 16:01:59
FilePath: \Vehicle_Drl\env\lib\path.py
Description: 

'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

def plot_path(path_points, path_type,position=None, closest_point=None, show=False):
    """
    绘制三维折线路径和（可选）给定位置与最近点。

    参数:
    position -- 三维空间中的一个点，将标记在图上。
    closest_point -- 给定位置的最近点，如果提供，也将标记在图上。
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(path_points[:, 0], path_points[:, 1], path_points[:, 2], label=path_type)

    if position is not None:
        ax.scatter([position[0]], [position[1]], [position[2]], color='r', label='Position')
    
    if closest_point is not None:
        ax.scatter([closest_point[0]], [closest_point[1]], [closest_point[2]], color='g', label='Closest Point')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend()
    if show:
        plt.show()


class Path:
    def __init__(self):
        pass
        
    def generate_path(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_closest_point(self, position):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_path_length(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    def plot_path(self):
        raise NotImplementedError("Subclasses should implement this!")
    
        

class PolylinePath(Path):
    def __init__(self, points):
        """
        初始化3D折线路径。

        参数:
        points -- 路径上的点，是一个形状为(N, 3)的数组，N是点的数量。
        """
        self.points = np.array(points)

    def generate_path(self):
        """
        生成折线路径上的所有点。

        返回值:
        路径上的点，是一个形状为(N, 3)的数组，N是点的数量。
        """
        return self.points

    def get_closest_point(self, position):
        """
        找到给定位置到折线路径最近的点。

        参数:
        position -- 一个形状为(3,)的数组，表示三维空间中的一个点。

        返回值:
        closest_point -- 路径上到给定位置最近的点。
        distance -- 给定位置到路径的最短距离。
        """
        closest_point = None
        min_distance = np.inf
        for i in range(len(self.points) - 1):
            segment_start = self.points[i]
            segment_end = self.points[i + 1]
            
            # 计算投影点
            vec = segment_end - segment_start
            vec_norm = vec / np.linalg.norm(vec)
            proj_length = np.dot((position - segment_start), vec_norm)
            proj_point = segment_start + proj_length * vec_norm
            
            # 确保投影点在线段上
            if proj_length < 0:
                closest_point_on_segment = segment_start
            elif proj_length > np.linalg.norm(vec):
                closest_point_on_segment = segment_end
            else:
                closest_point_on_segment = proj_point
            
            distance = np.linalg.norm(position - closest_point_on_segment)
            if distance < min_distance:
                min_distance = distance
                closest_point = closest_point_on_segment
        
        return closest_point, min_distance
    
    def get_path_length(self):
        """
        计算折线路径的总长度。

        返回值:
        路径的长度。
        """
        length = 0
        for i in range(len(self.points) - 1):
            length += np.linalg.norm(self.points[i + 1] - self.points[i])
        return length
    
    def plot_path(self, position=None, closest_point=None, show=False):
        """
        绘制三维折线路径和（可选）给定位置与最近点。

        参数:
        position -- 三维空间中的一个点，将标记在图上。
        closest_point -- 给定位置的最近点，如果提供，也将标记在图上。
        """
        plot_path(self.points, 'Polyline Path', position, closest_point, show)


class SplinePath(Path):
    def __init__(self, control_points):
        super().__init__()
        self.control_points = np.array(control_points)
        self.t_values = np.linspace(0, 1, len(control_points))
        self.spline_x = CubicSpline(self.t_values, self.control_points[:, 0])
        self.spline_y = CubicSpline(self.t_values, self.control_points[:, 1])
        self.spline_z = CubicSpline(self.t_values, self.control_points[:, 2])
        self.path_points = self.generate_path()
        
    def generate_path(self, num_points=100):
        t_new = np.linspace(0, 1, num_points)
        x_new = self.spline_x(t_new)
        y_new = self.spline_y(t_new)
        z_new = self.spline_z(t_new)
        return np.vstack((x_new, y_new, z_new)).T
    
    def distance_to_path(self, t, position):
        # 使用样条曲线上的点和给定点之间的欧氏距离
        point_on_spline = np.array([self.spline_x(t), self.spline_y(t), self.spline_z(t)])
        return np.linalg.norm(point_on_spline - position)
        
    def get_closest_point(self, position):
        # 最小化从给定点到样条曲线上的点的距离
        res = minimize_scalar(lambda t: self.distance_to_path(t, position), bounds=(0, 1), method='bounded')

        t_closest = res.x
        closest_point = np.array([self.spline_x(t_closest), self.spline_y(t_closest), self.spline_z(t_closest)])
        return closest_point, res.fun
    
    def get_path_length(self):
        path_points = self.generate_path(500)
        return np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
    
    def plot_path(self, position=None, closest_point=None, show=False):
        """
        绘制样条曲线路径和（可选）给定位置与最近点。

        参数:
        position -- 三维空间中的一个点，将标记在图上。
        closest_point -- 给定位置的最近点，如果提供，也将标记在图上。
        """
        plot_path(self.path_points, 'Polyline Path', position, closest_point, show)

    
    
class SineWavePath3D(Path):
    def __init__(self, amplitude, frequency, length, num_points=100):
        self.amplitude = amplitude
        self.frequency = frequency
        self.length = length
        self.num_points = num_points
        self.path_points = self.generate_path()

    def generate_path(self):
        """
        生成基于正弦函数的路径点。
        
        返回值:
        路径上的点，是一个形状为(num_points, 3)的数组。
        """
        t = np.linspace(0, self.length, self.num_points)
        x = t
        y = self.amplitude * np.sin(2 * np.pi * self.frequency * t)
        z = self.amplitude * np.cos(2 * np.pi * self.frequency * t)
        return np.vstack((x, y, z)).T
    
    def plot_path(self, position=None, closest_point=None, show=True):
        """
        绘制样条曲线路径和（可选）给定位置与最近点。

        参数:
        position -- 三维空间中的一个点，将标记在图上。
        closest_point -- 给定位置的最近点，如果提供，也将标记在图上。
        """
        plot_path(self.path_points,  'sineWave Path', position, closest_point, show)
        
    def distance_from_path(self, t, position):
        point_on_path = self.amplitude * np.array([t, np.sin(2 * np.pi * self.frequency * t), 
                                                    np.cos(2 * np.pi * self.frequency * t)])
        return np.linalg.norm(position - point_on_path)
        
    def get_closest_point(self, position):
        """
        寻找给定位置最近的路径点。

        参数:
        position -- 三维空间中的一个点（数组）

        返回值:
        路径上最近点的坐标。
        """
        # 使用数值方法寻找最小化位置和路径点距离的参数值
        
        result = minimize_scalar(lambda t: self.distance_from_path(t, position), bounds=(0, self.length), method='bounded')
        closest_t = result.x
        closest_point = np.array([closest_t, self.amplitude * np.sin(2 * np.pi * self.frequency * closest_t), 
                                                    self.amplitude * np.cos(2 * np.pi * self.frequency * closest_t)])
        return closest_point, result.fun
    

if __name__ == '__main__':
    # points = [[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 3, 3]]
    # polyline3D = PolylinePath(points)
    # position = [1, 2, 1]
    # closest_point = polyline3D.get_closest_point(position)
    # polyline3D.plot_path(position, closest_point)
    # print() 
    
    # spline = SplinePath([[0, 0, 0], [1, 1, 1], [2, 0, 2], [3, 3, 3]])
    # position = [1, 2, 1]
    # print(spline.get_path_length())
    # closest_point, distance = spline.get_closest_point(position)
    # print(distance)
    # spline.plot_path(position, closest_point, show=True)
    
    sine_wave_path = SineWavePath3D(amplitude=10, frequency=1, length=2)
    position = [1, 2, 1]
    # print(sine_wave_path.get_path_length())
    closest_point, distance = sine_wave_path.get_closest_point(position)
    print(distance)
    sine_wave_path.plot_path(position, closest_point, show=True)
