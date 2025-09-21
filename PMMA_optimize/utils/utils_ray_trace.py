import json
import math
from abc import abstractmethod
from enum import Enum
import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline


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
    def __init__(self, up_y_coords, down_y_coords, bound, num_control_points_up, num_control_points_down, **kwargs):
        """
        使用三次样条插值来定义和计算光学器件表面。
        """
        super().__init__(**kwargs)
        self.n1, self.n2, self.n3 = 1.0, 1.49, 1.0
        self.bound = bound
        
        # 1. 分别为上下表面定义控制点的 x 坐标
        # 确保 control_x_up 的长度与 up_y_coords 匹配
        self.control_x_up = np.linspace(bound[0], bound[1], num_control_points_up)
        
        # 确保 control_x_down 的长度与 down_y_coords 匹配
        self.control_x_down = np.linspace(bound[0], bound[1], num_control_points_down)

        # 2. 使用匹配的 x, y 坐标创建三次样条插值器
        self.up_spline = CubicSpline(self.control_x_up, up_y_coords)
        self.down_spline = CubicSpline(self.control_x_down, down_y_coords)

    def up_surface_fun(self, x):
        """使用样条插值器计算上表面 y 坐标。"""
        return self.up_spline(x)

    def down_surface_fun(self, x):
        """使用样条插值器计算下表面 y 坐标。"""
        return self.down_spline(x)

    def up_surface_normal(self, x):
        """计算上表面法线（指向+y方向）。"""
        # CubicSpline 对象可以直接计算导数 (nu=1)
        dy_dx = self.up_spline(x, nu=1)
        normal = np.vstack((-dy_dx, np.ones_like(x)))
        return normal / np.linalg.norm(normal, axis=0)

    def down_surface_normal(self, x):
        """计算下表面法线（指向+y方向）。"""
        dy_dx = self.down_spline(x, nu=1)
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
        """
        Traces rays through a 2D lens and returns the points at each stage.
        
        Modified Logic: This version retains all rays that successfully intersect the first
        surface. If a ray fails to intersect the second surface, its final state is
        calculated by extending it a fixed distance forward.
        """
        # 1. Coordinate Transformation
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, is_vector=True)
        
        # 2. Intersection with the lower surface
        t1 = self._find_intersection_newton(pl0_bcs, n1_bcs, self.down_surface_fun, lambda x: self.up_spline(x, nu=1))
        pl1_bcs = pl0_bcs + t1 * n1_bcs
        
        # 3. First Filtering: Based on boundary of the lower surface
        pl1_filtered, (pl0_filtered, n1_filtered,), mask1 = filter_rays_by_boundary(
            pl1_bcs, (self.bound[0], self.bound[1]), pl0_bcs, n1_bcs
        )
        if pl1_filtered.shape[1] == 0:
            return [np.empty((2, 0))] * 4 # Return 4 empty arrays

        # 4. Refraction at the lower surface
        normal1 = self.down_surface_normal(pl1_filtered[0, :])
        n2_refracted = calculate_refraction_vector(self.n1, n1_filtered, self.n2, normal1)
        
        # 5. Second Filtering: Based on Total Internal Reflection (TIR)
        mask_no_tir1 = ~np.isnan(n2_refracted[0, :])
        pl0_valid = pl0_filtered[:, mask_no_tir1]
        pl1_valid = pl1_filtered[:, mask_no_tir1]
        n2_valid = n2_refracted[:, mask_no_tir1]
        
        # This is the set of all rays that have successfully entered the lens
        if pl1_valid.shape[1] == 0:
            return [np.empty((2, 0))] * 4

        # ====================================================================
        # ## Core Modification Start ##
        # ====================================================================

        # 6. Calculate intersection with the upper surface for ALL valid rays
        t2 = self._find_intersection_newton(pl1_valid, n2_valid, self.up_surface_fun, lambda x: self.up_spline(x, nu=1))
        pl2_intersections = pl1_valid + t2 * n2_valid

        # 7. "Classify" rays instead of filtering: find which rays hit the upper surface boundary
        _, _, mask_hit_up_surface = filter_rays_by_boundary(
            pl2_intersections, (self.bound[0], self.bound[1])
        )

        # Prepare final arrays to hold results for both "hit" and "miss" cases
        p2_final_bcs = np.zeros_like(pl1_valid)
        n3_final_bcs = np.zeros_like(n2_valid)

        # 8. Process rays that successfully hit the second surface
        if np.any(mask_hit_up_surface):
            # Select the rays that hit
            pl2_hit = pl2_intersections[:, mask_hit_up_surface]
            n2_hit = n2_valid[:, mask_hit_up_surface]
            
            # Perform reflection on the upper surface
            normal2 = self.up_surface_normal(pl2_hit[0, :])
            n3_hit = calculate_reflection_vector(n2_hit, normal2)
            
            # Place results into their corresponding positions in the final arrays
            p2_final_bcs[:, mask_hit_up_surface] = pl2_hit
            n3_final_bcs[:, mask_hit_up_surface] = n3_hit

        # 9. Process rays that missed the second surface
        mask_miss_up_surface = ~mask_hit_up_surface
        if np.any(mask_miss_up_surface):
            # Select the rays that missed
            pl1_miss = pl1_valid[:, mask_miss_up_surface]
            n2_miss = n2_valid[:, mask_miss_up_surface]
            
            # Apply the new rule: p2 = p1 + 100 * n2, n3 = n2
            p2_for_missed = pl1_miss + 100 * n2_miss
            n3_for_missed = n2_miss
            
            # Place results into their corresponding positions in the final arrays
            p2_final_bcs[:, mask_miss_up_surface] = p2_for_missed
            n3_final_bcs[:, mask_miss_up_surface] = n3_for_missed
            
        # ====================================================================
        # ## Core Modification End ##
        # ====================================================================

        # 10. Convert all aligned arrays back to world coordinates and return
        # At this point, p0_valid, p1_valid, p2_final_bcs, and n3_final_bcs are all
        # correctly sized and aligned.
        p0_wcs = self.OptEl_coordinate_to_world_coordinate(pl0_valid)
        p1_wcs = self.OptEl_coordinate_to_world_coordinate(pl1_valid)
        p2_wcs = self.OptEl_coordinate_to_world_coordinate(p2_final_bcs)
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_final_bcs, is_vector=True)
        
        return p0_wcs, p1_wcs, p2_wcs, n3_wcs

    def plot_element_2d(self, ax, **kwargs):
        """在2D坐标轴上绘制透镜轮廓。"""
        x_local = np.linspace(self.bound[0], self.bound[1], 200)
        up_y_local = self.up_surface_fun(x_local)
        down_y_local = self.down_surface_fun(x_local)
        
        up_pts = self.OptEl_coordinate_to_world_coordinate(np.vstack([x_local, up_y_local]))
        down_pts = self.OptEl_coordinate_to_world_coordinate(np.vstack([x_local, down_y_local]))
        
        all_x = np.concatenate([up_pts[0], down_pts[0, ::-1], [up_pts[0, 0]]])
        all_y = np.concatenate([up_pts[1], down_pts[1, ::-1], [up_pts[1, 0]]])
        
        ax.fill(all_x, all_y, alpha=0.2, **kwargs)
        ax.plot(all_x, all_y, **kwargs)


class Point_light_source(OptElement):
    """
    2D点光源，其辐射方向根据给定的强度曲线进行加权。
    """
    def __init__(self, num_rays=1000, radiation_profile=None, **kwargs):
        """
        初始化点光源。
        
        :param num_rays: 生成的光线数量。
        :param radiation_profile: 光源的辐射特性曲线，一个 Nx2 的数组，
                                  格式为 [[角度1, 强度1], [角度2, 强度2], ...]。
                                  角度从主轴（0度）开始计算。
                                  如果为 None，则使用用户指定的默认分布。
        :param kwargs: 传递给 OptElement 基类的参数（例如位姿矩阵）。
        """
        super().__init__(**kwargs)
        self.num_rays = num_rays
        
        if radiation_profile is None:
            # --- 核心修改部分 ---
            # 如果未提供特性曲线，则默认使用您指定的新点对
            self.radiation_profile = np.array([
                [0, 1.0], [10, 0.98], [20, 0.96], [30, 0.90], [40, 0.82],
                [50, 0.70], [60, 0.55], [70, 0.37], [80, 0.12], [90, 0.00]
            ])
            # --- 修改结束 ---
        else:
            self.radiation_profile = np.array(radiation_profile)
            
        # 预计算累积分布函数 (CDF) 以便快速采样
        self._prepare_cdf()

    def _prepare_cdf(self):
        """
        根据辐射曲线计算累积分布函数 (CDF)，用于逆变换采样。
        【已修改】根据数据点数量动态选择插值方法，避免错误。
        """
        angles_deg_orig = self.radiation_profile[:, 0]
        intensities_orig = self.radiation_profile[:, 1]

        # 使用插值来创建更平滑、更密集的曲线
        angles_deg_interp = np.linspace(angles_deg_orig.min(), angles_deg_orig.max(), 1000)
        
        num_points = len(angles_deg_orig)
        
        if num_points >= 4:
            interp_kind = 'cubic'
        elif num_points == 3:
            interp_kind = 'quadratic'
        else: 
            interp_kind = 'linear'
        
        if num_points > 1:
            f_intensities = interp1d(angles_deg_orig, intensities_orig, kind=interp_kind, bounds_error=False, fill_value=0)
            intensities_interp = f_intensities(angles_deg_interp)
        else:
            intensities_interp = np.full_like(angles_deg_interp, intensities_orig[0])

        intensities_interp[intensities_interp < 0] = 0
        self.angles_rad_interp = np.deg2rad(angles_deg_interp)

        cdf_unnormalized = np.cumsum(intensities_interp)
        
        if cdf_unnormalized[-1] > 0:
            self.cdf_normalized = cdf_unnormalized / cdf_unnormalized[-1]
        else: 
            self.cdf_normalized = np.linspace(0, 1, len(angles_deg_interp))


    def trace_ray(self):
        """
        使用逆变换采样生成光线，使其方向符合辐射特性曲线。
        """
        p_local = np.zeros((2, self.num_rays))
        np.random.seed(0)  # 确保结果可复现
        uniform_samples = np.random.rand(self.num_rays)
        sampled_angles = np.interp(uniform_samples, self.cdf_normalized, self.angles_rad_interp)
        signs = np.random.choice([-1, 1], self.num_rays)
        final_angles = sampled_angles * signs
        
        n_local = np.vstack((np.sin(final_angles), np.cos(final_angles)))
        
        p_wcs = self.OptEl_coordinate_to_world_coordinate(p_local)
        n_wcs = self.OptEl_coordinate_to_world_coordinate(n_local, is_vector=True)
        return p_wcs, n_wcs

    def plot_element_2d(self, ax, **kwargs):
        """在2D坐标轴上绘制光源位置和方向箭头。"""
        # 1. 获取光源在世界坐标系中的位置和旋转矩阵
        pos = self.OptEl_to_world_translation_matrix
        # rot_matrix = self.OptEl_to_world_rotation_matrix # 旋转矩阵在 OptEl_coordinate_to_world_coordinate 内部使用

        # 2. 绘制光源位置的标记 (您的原始功能，保持不变)
        marker_color = kwargs.pop('color', 'yellow')
        marker_label = kwargs.pop('label', 'source')
        
        # 从 (2,1) 或 (1,2) 的位置矩阵中提取标量 x, y
        start_x = pos.flatten()[0]
        start_y = pos.flatten()[1]

        ax.plot(start_x, start_y, marker='*', markersize=15, color=marker_color, label=marker_label, **kwargs)

        # 3. 计算并绘制方向箭头
        
        # 定义一个在光源本地坐标系下的“朝上”向量 (0, 1)
        # 这是光源自身的“前进”方向
        local_direction_vector = np.array([[0], [1]])

        # 使用您的转换函数，将这个本地向量转换为世界坐标系中的方向向量
        # is_vector=True 确保只应用旋转，不应用平移
        world_direction_vector = self.OptEl_coordinate_to_world_coordinate(
            local_direction_vector, is_vector=True
        ).flatten()

        # 为了可视化，给箭头一个固定的长度
        arrow_length = 15.0  # 您可以根据坐标轴的范围调整这个值

        # 计算箭头在世界坐标系中的 x 和 y 方向分量 (dx, dy)
        dx = arrow_length * world_direction_vector[0]
        dy = arrow_length * world_direction_vector[1]

        # 直接使用计算出的 start_x, start_y 和 dx, dy 来绘制箭头
        ax.arrow(
            start_x,
            start_y,
            dx,
            dy,
            head_width=3,      # 箭头头部的宽度
            head_length=4,     # 箭头头部的长度
            fc='red',          # 箭头的填充颜色
            ec='red',          # 箭头的边框颜色
            label='vector',
            length_includes_head=True # 使箭头总长接近 arrow_length
        )

