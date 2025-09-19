import json
import math
from abc import abstractmethod
from enum import Enum
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R

# =============================================================================
# 1. 2D旋转与坐标变换辅助函数
# =============================================================================

def is_pure_rotation(matrix):
    """检查2x2矩阵是否为纯旋转矩阵（行列式为1）。"""
    return np.isclose(np.linalg.det(matrix), 1.0)

def rotvec_to_matrix(angle, reflection_matrix=None):
    """将2D旋转角度（和可选的反射矩阵）转换为2x2矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    rotation_matrix = np.array([[c, -s], [s, c]])
    if reflection_matrix is not None:
        return rotation_matrix @ reflection_matrix
    return rotation_matrix

def matrix_to_rotvec(matrix):
    """将任意2x2正交矩阵转换为旋转角度和可选的反射矩阵。"""
    if is_pure_rotation(matrix):
        angle = np.arctan2(matrix[1, 0], matrix[0, 0])
        return angle, None
    else:
        # 假设一个标准的反射矩阵（例如沿x轴反射）
        reflection_matrix = np.array([[1, 0], [0, -1]])
        # 消除反射效应以提取纯旋转
        rotation_matrix = matrix @ reflection_matrix
        angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        return angle, reflection_matrix

# =============================================================================
# 2. 枚举类与几何计算辅助函数
# =============================================================================

class TransformOrder(Enum):
    """坐标变换方法，先平移后旋转还是先旋转后平移。"""
    TRANSLATE_ROTATE = 1
    ROTATE_TRANSLATE = 2

def calculate_reflection_vector(incident_direction, normal_direction):
    """根据入射光线和法线计算反射光线的2D方向向量（向量化）。"""
    incident_norm = incident_direction / np.linalg.norm(incident_direction, axis=0, keepdims=True)
    normal_norm = normal_direction / np.linalg.norm(normal_direction, axis=0, keepdims=True)
    dot_product = np.sum(incident_norm * normal_norm, axis=0)
    reflection_vector = incident_norm - 2 * dot_product * normal_norm
    return reflection_vector / np.linalg.norm(reflection_vector, axis=0, keepdims=True)

def calculate_refraction_vector(n_1, incident_direction, n_2, normal_direction):
    """计算2D折射方向向量（向量化），能处理全内反射。"""
    incident_dir = incident_direction / np.linalg.norm(incident_direction, axis=0, keepdims=True)
    normal_dir = normal_direction / np.linalg.norm(normal_direction, axis=0, keepdims=True)
    
    cos_i = np.sum(incident_dir * normal_dir, axis=0)
    
    # 保证法线与入射光线方向相反
    normal_adj = np.copy(normal_dir)
    flip_mask = cos_i > 0
    normal_adj[:, flip_mask] *= -1
    cos_i[flip_mask] *= -1
    
    n_ratio = n_1 / n_2
    radicand = 1.0 - n_ratio**2 * (1.0 - cos_i**2)
    
    # 处理全内反射
    tir_mask = radicand < 0
    refracted_vector = np.full_like(incident_dir, np.nan)
    
    valid_mask = ~tir_mask
    if np.any(valid_mask):
        cos_t = np.sqrt(radicand[valid_mask])
        refracted_vector[:, valid_mask] = n_ratio * incident_dir[:, valid_mask] + (n_ratio * (-cos_i[valid_mask]) - cos_t) * normal_adj[:, valid_mask]

    return refracted_vector

def filter_rays_by_boundary(points_to_check, boundary_x, *arrays_to_filter):
    """根据一维边界筛选光线数据。"""
    if points_to_check.shape[1] == 0:
        return np.empty((2,0)), tuple(arr.reshape(arr.shape[0], 0) for arr in arrays_to_filter), np.array([], dtype=bool)

    mask = (points_to_check[0, :] >= boundary_x[0]) & (points_to_check[0, :] <= boundary_x[1])
    points_inside = points_to_check[:, mask]
    filtered_arrays = tuple(arr[:, mask] for arr in arrays_to_filter)
    return points_inside, filtered_arrays, mask

def plot_2d_rays(ax, start_points, end_points, num_to_plot=50, **kwargs):
    """高效绘制2D光线路径。"""
    num_rays = start_points.shape[1]
    if num_rays == 0: return
    
    indices = np.random.choice(num_rays, min(num_rays, num_to_plot), replace=False)
    
    segments = np.array([start_points[:, indices].T, end_points[:, indices].T]).transpose((1, 0, 2))
    line_collection = LineCollection(segments, **kwargs)
    ax.add_collection(line_collection)

# =============================================================================
# 3. 核心光学元件类
# =============================================================================

class OptElement:
    """光学元件的2D基类。"""
    def __init__(self, OptEl_to_world_translation_matrix=np.zeros((2,1)),
                 OptEl_to_world_rotation_matrix=np.eye(2)):
        self.set_pose(OptEl_to_world_rotation_matrix, OptEl_to_world_translation_matrix)
        self.world_to_OptEl_angle, self.world_to_OptEl_reflection = matrix_to_rotvec(self.world_to_OptEl_rotation_matrix)

    def set_pose(self, rotation_matrix, translation_vector):
        """设置光学元件在世界坐标系中的位姿。"""
        self.OptEl_to_world_rotation_matrix = np.array(rotation_matrix)
        self.OptEl_to_world_translation_matrix = np.array(translation_vector).reshape(2, 1)
        # 逆变换：p_local = R.T @ (p_world - T)
        self.world_to_OptEl_rotation_matrix = self.OptEl_to_world_rotation_matrix.T

    def get_parameters(self):
        """获取旋转角度和平移向量。"""
        return self.world_to_OptEl_angle, self.OptEl_to_world_translation_matrix

    def update_parameters(self, angle, trans_vec):
        """通过角度和向量更新位姿。"""
        rot_mat = rotvec_to_matrix(angle, self.world_to_OptEl_reflection)
        self.set_pose(rot_mat, trans_vec)

    def OptEl_coordinate_to_world_coordinate(self, p_local, is_vector=False):
        """本地坐标 -> 世界坐标。"""
        if is_vector:
            return self.OptEl_to_world_rotation_matrix @ p_local
        else:
            return self.OptEl_to_world_rotation_matrix @ p_local + self.OptEl_to_world_translation_matrix

    def world_coordinate_to_OptEl_coordinate(self, p_world, is_vector=False):
        """世界坐标 -> 本地坐标。"""
        if is_vector:
            return self.world_to_OptEl_rotation_matrix @ p_world
        else:
            return self.world_to_OptEl_rotation_matrix @ (p_world - self.OptEl_to_world_translation_matrix)
    
    @abstractmethod
    def trace_ray(self, p_wcs, n_wcs):
        pass

    def plot_element_2d(self, ax, **kwargs):
        pass

class LC_device(OptElement):
    """
    2D自由曲面透镜类，使用牛顿法进行快速光线-曲线求交。
    """
    def __init__(self, up_surface_params, down_surface_params, bound_x=11.2, **kwargs):
        super().__init__(**kwargs)
        self.n1, self.n2, self.n3 = 1.0, 1.49, 1.0
        self.up_surface_params = np.array(up_surface_params)
        self.down_surface_params = np.array(down_surface_params)
        self.bound_x = bound_x
    
    def _poly_calc(self, x, params):
        """计算多项式 y = c0 + c1*x + c2*x^2 + ..."""
        x_powers = np.vander(x, len(params), increasing=True)
        return x_powers @ params

    def _poly_derivative(self, x, params):
        """计算多项式导数 dy/dx。"""
        if len(params) < 2:
            return np.zeros_like(x)
        der_params = np.array([i * p for i, p in enumerate(params) if i > 0])
        x_powers = np.vander(x, len(der_params), increasing=True)
        return x_powers @ der_params
    
    def up_surface_fun(self, x): return self._poly_calc(x, self.up_surface_params)
    def down_surface_fun(self, x): return self._poly_calc(x, self.down_surface_params)
    
    def up_surface_normal(self, x):
        """计算上表面法线（指向+y方向）。"""
        dy_dx = self._poly_derivative(x, self.up_surface_params)
        normal = np.vstack((-dy_dx, np.ones_like(x)))
        return normal / np.linalg.norm(normal, axis=0)

    def down_surface_normal(self, x):
        """计算下表面法线（指向+y方向）。"""
        dy_dx = self._poly_derivative(x, self.down_surface_params)
        normal = np.vstack((-dy_dx, np.ones_like(x)))
        return normal / np.linalg.norm(normal, axis=0)

    def _find_intersection_newton(self, ray_origin, ray_dir, surface_func, derivative_func):
        """使用牛顿法快速求解光线与曲线的交点参数 t。"""
        t = np.ones(ray_origin.shape[1]) * 5.0 # 初始猜测
        for _ in range(10): # 牛顿法迭代
            x_intersect = ray_origin[0, :] + t * ray_dir[0, :]
            y_ray = ray_origin[1, :] + t * ray_dir[1, :]
            y_surface = surface_func(x_intersect)
            
            f_t = y_ray - y_surface
            
            dy_dx = derivative_func(x_intersect)
            f_prime_t = ray_dir[1, :] - dy_dx * ray_dir[0, :]
            
            # 防止除零
            f_prime_t[np.abs(f_prime_t) < 1e-9] = 1e-9
            
            delta_t = -f_t / f_prime_t
            t += delta_t
            
            if np.all(np.abs(delta_t) < 1e-7): break
        return t

    def trace_ray(self, p_wcs, n_wcs):
        """追踪光线穿过2D透镜，并返回各阶段对应的点。"""
        # 1. 坐标转换
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, is_vector=True)
        
        # 2. 与下表面求交
        t1 = self._find_intersection_newton(pl0_bcs, n1_bcs, self.down_surface_fun, lambda x: self._poly_derivative(x, self.down_surface_params))
        pl1_bcs = pl0_bcs + t1 * n1_bcs
        
        # 3. 第一次筛选：基于边界
        # 此时，pl0_bcs 和 n1_bcs 仍然是全部的 10000 条光线
        pl1_filtered, (pl0_filtered, n1_filtered,), mask1 = filter_rays_by_boundary(
            pl1_bcs, (-self.bound_x/2, self.bound_x/2), pl0_bcs, n1_bcs
        )
        if pl1_filtered.shape[1] == 0:
            return [np.empty((2,0))] * 5 # 返回5个空数组

        # 4. 在下表面折射
        normal1 = self.down_surface_normal(pl1_filtered[0, :])
        n2_refracted = calculate_refraction_vector(self.n1, n1_filtered, self.n2, normal1)
        
        # 5. 第二次筛选：基于是否发生全内反射 (TIR)
        mask_no_tir1 = ~np.isnan(n2_refracted[0, :])
        pl0_after_tir_filter = pl0_filtered[:, mask_no_tir1]
        pl1_after_tir_filter = pl1_filtered[:, mask_no_tir1]
        n2_after_tir_filter = n2_refracted[:, mask_no_tir1]
        if pl1_after_tir_filter.shape[1] == 0:
            return [np.empty((2,0))] * 5

        # 6. 与上表面求交
        t2 = self._find_intersection_newton(pl1_after_tir_filter, n2_after_tir_filter, self.up_surface_fun, lambda x: self._poly_derivative(x, self.up_surface_params))
        pl2_bcs = pl1_after_tir_filter + t2 * n2_after_tir_filter

        # 7. 第三次筛选：基于上表面边界
        pl2_filtered, (pl1_final_candidates, n2_filtered, pl0_final_candidates,), mask2 = filter_rays_by_boundary(
            pl2_bcs, (-self.bound_x/2, self.bound_x/2), pl1_after_tir_filter, n2_after_tir_filter, pl0_after_tir_filter
        )
        if pl2_filtered.shape[1] == 0:
            return [np.empty((2,0))] * 5

        # 8. 在上表面折射/反射
        normal2 = self.up_surface_normal(pl2_filtered[0, :])
        n3_final_candidates = calculate_reflection_vector(n2_filtered,normal2)
        
        # 9. 第四次筛选：基于第二次TIR
        mask_final = ~np.isnan(n3_final_candidates[0, :])
        
        # 应用最终筛选到所有数组，确保它们一一对应
        p0_final = pl0_final_candidates[:, mask_final]
        p1_final = pl1_final_candidates[:, mask_final]
        p2_final = pl2_filtered[:, mask_final]
        n3_final = n3_final_candidates[:, mask_final]
        
        # 10. 转换回世界坐标并返回
        p0_wcs = self.OptEl_coordinate_to_world_coordinate(p0_final)
        p1_wcs = self.OptEl_coordinate_to_world_coordinate(p1_final)
        p2_wcs = self.OptEl_coordinate_to_world_coordinate(p2_final)
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_final, is_vector=True)
        
        return p0_wcs, p1_wcs, p2_wcs, n3_wcs

    def plot_element_2d(self, ax, **kwargs):
        """在2D坐标轴上绘制透镜轮廓。"""
        x_local = np.linspace(-self.bound_x/2, self.bound_x/2, 200)
        up_y_local = self.up_surface_fun(x_local)
        down_y_local = self.down_surface_fun(x_local)
        
        up_pts = self.OptEl_coordinate_to_world_coordinate(np.vstack([x_local, up_y_local]))
        down_pts = self.OptEl_coordinate_to_world_coordinate(np.vstack([x_local, down_y_local]))
        
        all_x = np.concatenate([up_pts[0], down_pts[0, ::-1], [up_pts[0, 0]]])
        all_y = np.concatenate([up_pts[1], down_pts[1, ::-1], [up_pts[1, 0]]])
        
        ax.fill(all_x, all_y, alpha=0.2, **kwargs)
        ax.plot(all_x, all_y, **kwargs)


class Point_light_source(OptElement):
    """2D点光源，在扇形区域内均匀发射光线。"""
    def __init__(self, num_rays=1000, emission_angle_deg=45, **kwargs):
        super().__init__(**kwargs)
        self.num_rays = num_rays
        self.emission_angle_rad = np.deg2rad(emission_angle_deg)
        
    def trace_ray(self):
        p_local = np.zeros((2, self.num_rays))
        
        # 在 [-angle/2, +angle/2] 范围内均匀生成角度
        angles = np.random.uniform(-self.emission_angle_rad / 2, self.emission_angle_rad / 2, self.num_rays)
        
        # 角度转为方向向量 (本地坐标系，主方向为+y)
        n_local = np.vstack((np.sin(angles), np.cos(angles)))
        
        # 转换到世界坐标系
        p_wcs = self.OptEl_coordinate_to_world_coordinate(p_local)
        n_wcs = self.OptEl_coordinate_to_world_coordinate(n_local, is_vector=True)
        return p_wcs, n_wcs

    def plot_element_2d(self, ax, **kwargs):
        """在2D坐标轴上绘制光源位置。"""
        pos = self.OptEl_to_world_translation_matrix
        ax.plot(pos[0], pos[1], marker='*', markersize=12, color='yellow', label='光源', **kwargs)

# =============================================================================
# 4. 示例与可视化
# =============================================================================
if __name__ == '__main__':
    # --- 1. 初始化光学系统 ---
    
    # 定义光源：10000条光线，发射角为60度
    light_source = Point_light_source(num_rays=10000, emission_angle_deg=60)
    
    # 定义透镜：上下表面由4个参数的多项式定义 y = c0 + c1*x + c2*x^2 + c3*x^3
    # 初始形状为一个简单的平凸透镜
    up_params = [39, -1, 0, 0]  # 平面 y = 1.5
    down_params = [0, 0,0, 0] # 曲面 y = -0.05*x^2
    lens = LC_device(
        up_surface_params=up_params,
        down_surface_params=down_params,
        bound_x=20.0,
        OptEl_to_world_translation_matrix=np.array([[0], [5]])
    )
    
    # --- 2. 执行光线追迹 ---
    print("正在执行光线追迹...")
    initial_p_all, initial_n_all = light_source.trace_ray()
    
    # 修改这里的解包，以接收新的返回值
    # p0_final 是与 p1_final, p2_final 一一对应的初始点
    p0_final, p1_final, p2_final, n_final = lens.trace_ray(initial_p_all, initial_n_all)
    print(f"共 {initial_p_all.shape[1]} 条初始光线，成功追迹 {p0_final.shape[1]} 条光线。")

    # --- 3. 可视化结果 ---
    print("正在生成可视化图像...")
    fig, ax = plt.subplots(figsize=(10, 10))
    # ... (其他绘图设置不变) ...

    # 绘制光源和透镜
    light_source.plot_element_2d(ax)
    lens.plot_element_2d(ax, color='cyan', label='透镜')

    # 绘制光线路径
    # 现在 p0_final 和 p1_final 的长度是匹配的！
    plot_2d_rays(ax, p0_final, p1_final, num_to_plot=100, colors='orange', linewidths=0.3, alpha=0.5, label='入射光线')
    
    # p1_final 和 p2_final 的长度也是匹配的！
    plot_2d_rays(ax, p1_final, p2_final, num_to_plot=100, colors='deepskyblue', linewidths=0.3, alpha=0.7, label='内部光线')
    
    p_final_extended = p2_final + n_final * 20
    plot_2d_rays(ax, p2_final, p_final_extended, num_to_plot=100, colors='lime', linewidths=0.5, alpha=0.8, label='出射光线')

    ax.legend()
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.show()