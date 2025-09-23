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
from stl import mesh


# =============================================================================
# 1. 2D旋转与坐标变换辅助函数
# =============================================================================
def generate_device_stl_from_npz(npz_path, stl_path, x_range, y_range, resolution=50):
    """
    从.npz文件中读取上下表面参数，生成一个三维实体STL文件，并计算光源的世界坐标。

    :param npz_path: 输入的 .npz 文件路径。
    :param stl_path: 输出的 .stl 文件路径。
    :param x_range: 器件的X轴范围，格式为 (x_min, x_max)。
    :param y_range: 器件的Y轴范围 (宽度)，格式为 (y_min, y_max)。
    :param resolution: 曲面网格的精细度，数字越大，模型越精细。
    :return: light_sources_world_pos (list): 三个光源在3D世界坐标系中的位置列表。
    """
    print(f"--- 开始处理文件: {npz_path} ---")

    # --- 1. 加载数据 ---
    try:
        data = np.load(npz_path)
        up_y_coords = data['up_y_coords']
        down_y_coords = data['down_y_coords']
        num_up = data['num_up']
        num_down = data['num_down']
        light_params = data['light_params']
        print("✅ NPZ 文件加载成功。")
    except FileNotFoundError:
        print(f"❌ 错误: 文件 '{npz_path}' 未找到。")
        return None
    except KeyError as e:
        print(f"❌ 错误: NPZ 文件中缺少必要的键: {e}")
        return None

    # --- 2. 重建样条曲线 ---
    control_x_up = np.linspace(x_range[0], x_range[1], num_up)
    control_x_down = np.linspace(x_range[0], x_range[1], num_down)

    up_spline = CubicSpline(control_x_up, up_y_coords)
    down_spline = CubicSpline(control_x_down, down_y_coords)
    print("✅ 样条曲线重建完成。")

    # --- 3. 生成三维网格顶点 ---
    x_samples = np.linspace(x_range[0], x_range[1], resolution)
    y_samples = np.linspace(y_range[0], y_range[1], resolution)
    y_center = (y_range[0] + y_range[1]) / 2.0
    X, Y = np.meshgrid(x_samples, y_samples)

    # 计算上、下表面的 Z 坐标
    Z_up = up_spline(X)
    Z_down = down_spline(X)

    # 将所有顶点组合成一个列表
    # 顺序: 上表面顶点 -> 下表面顶点
    verts_up = np.stack([X.flatten(), Y.flatten(), Z_up.flatten()], axis=1)
    verts_down = np.stack([X.flatten(), Y.flatten(), Z_down.flatten()], axis=1)
    all_vertices = np.vstack([verts_up, verts_down])
    print("✅ 三维顶点生成完成。")

    # --- 4. 生成三角面片 ---
    faces = []
    res = resolution
    offset = res * res # 下表面顶点的起始索引

    for j in range(res - 1):
        for i in range(res - 1):
            # 当前网格单元的四个顶点索引
            p1 = j * res + i
            p2 = j * res + i + 1
            p3 = (j + 1) * res + i
            p4 = (j + 1) * res + i + 1
            
            # 上表面 (法线朝外, +z)
            faces.append([p1, p2, p4])
            faces.append([p1, p4, p3])
            
            # 下表面 (法线朝外, -z), 顶点顺序相反
            faces.append([offset + p1, offset + p4, offset + p2])
            faces.append([offset + p1, offset + p3, offset + p4])

    # 封闭四个侧面
    for i in range(res - 1):
        # 前侧面 (y = y_min)
        p1_up, p2_up = i, i + 1
        p1_down, p2_down = offset + i, offset + i + 1
        faces.append([p1_up, p2_down, p1_down])
        faces.append([p1_up, p2_up, p2_down])

        # 后侧面 (y = y_max)
        p1_up, p2_up = (res-1)*res + i, (res-1)*res + i + 1
        p1_down, p2_down = offset + (res-1)*res + i, offset + (res-1)*res + i + 1
        faces.append([p1_up, p1_down, p2_down])
        faces.append([p1_up, p2_down, p2_up])
        
        # 左侧面 (x = x_min)
        p1_up, p2_up = i*res, (i+1)*res
        p1_down, p2_down = offset + i*res, offset + (i+1)*res
        faces.append([p1_up, p1_down, p2_down])
        faces.append([p1_up, p2_down, p2_up])

        # 右侧面 (x = x_max)
        p1_up, p2_up = i*res + res-1, (i+1)*res + res-1
        p1_down, p2_down = offset + i*res + res-1, offset + (i+1)*res + res-1
        faces.append([p1_up, p2_down, p1_down])
        faces.append([p1_up, p2_up, p2_down])
        
    print("✅ 三角面片生成完成。")
    
    # --- 5. 创建并保存STL文件 ---
    device_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        device_mesh.vectors[i] = all_vertices[f]
        
    device_mesh.save(stl_path)
    print(f"💾 STL 文件已保存至: '{stl_path}'")
    
    # --- 6. 计算光源的三维世界坐标 ---
    
    # 计算光源相对于器件原点的局部2D坐标
    light_configs_local = calculate_light_sources_from_params(*light_params)
    
    light_sources_world_pos = []
    for cfg in light_configs_local:
        local_pos_2d = cfg['OptEl_to_world_translation_matrix'].flatten()
        
        # 转换为3D坐标 (假设器件和光源都在 z=0 平面) 
        # 注意: STL模型中的Z轴对应我们2D坐标系中的Y轴
        world_pos_3d = [local_pos_2d[0], y_center, local_pos_2d[1]]
        light_sources_world_pos.append(world_pos_3d)

    print("✅ 光源三维坐标计算完成。")
    
    return light_sources_world_pos
def calculate_light_sources_from_params(a, l1, l2, l3):
    """
    【2D版本】根据一个固定的y坐标'a'和三个相对位置参数，计算三个光源的位置和姿态。

    参数定义已更新:
    :param a: 光源所在水平线的 y 坐标值。
    :param l1: 中间光源在水平线段上的位置比例。范围[-1, 1]。
              -1代表左端点, 0代表中心点, 1代表右端点。
    :param l2: 左侧光源的位置比例。插值区间为 [距离中间光源1.8mm的左锚点] 到 [线段左端点]。范围[0, 1]。
    :param l3: 右侧光源的位置比例。插值区间为 [距离中间光源1.8mm的右锚点] 到 [线段右端点]。范围[0, 1]。
    :return: light_sources_config_2d 列表
    """
    # --- a. 定义光源所在的水平线段 ---
    # x 范围与之前的矩形边界保持一致
    x_bounds = {'x_min': 0, 'x_max': 8.3}
    
    # 线段的左右端点，y坐标由参数'a'直接决定
    p_start = np.array([x_bounds['x_min'], a])
    p_end = np.array([x_bounds['x_max'], a])
    
    # --- b. 计算线段中心和半长向量 ---
    segment_center = (p_start + p_end) / 2.0
    segment_half_vector = (p_end - p_start) / 2.0

    # ####################################################################
    # ## c. 计算光源位置 (基于新逻辑) ##
    # ####################################################################
    
    # 1. 根据 l1 计算中间光源的位置
    pos_middle = segment_center + l1 * segment_half_vector
    
    # 2. 计算左右两个新的“锚点”，它们是插值的起点
    # 因为是在水平线上，单位向量非常简单
    
    # 朝向左端点(p_start)的单位向量是 [-1, 0]
    p_left_anchor = pos_middle + 1.8 * np.array([-1.0, 0.0])

    # 朝向右端点(p_end)的单位向量是 [1, 0]
    p_right_anchor = pos_middle + 1.8 * np.array([1.0, 0.0])

    # 3. 根据 l2，在新的“左锚点”和“左端点(p_start)”之间进行线性插值
    pos_left = p_left_anchor + l2 * (p_start - p_left_anchor)
    
    # 4. 根据 l3，在新的“右锚点”和“右端点(p_end)”之间进行线性插值
    pos_right = p_right_anchor + l3 * (p_end - p_right_anchor)
    
    # 5. 组合并塑形
    positions_2d = [pos_left, pos_middle, pos_right]
    positions_2d_col = [pos.reshape(2, 1) for pos in positions_2d]

    # ####################################################################
    # ## d. 计算光源姿态 (旋转矩阵) ##
    # ####################################################################
    
    # 由于不再有斜率，我们假设光源始终指向上方 (+y方向)
    # 这是一个标准的、无旋转的姿态（单位矩阵）
    # 本地坐标系的 x'轴 -> 世界坐标系的 x轴 [1, 0]
    # 本地坐标系的 y'轴 -> 世界坐标系的 y轴 [0, 1]
    rotation_matrix_2d = np.eye(2)
    
    # ####################################################################
    
    # --- e. 构建2D光源配置列表 ---
    light_sources_config_2d = []
    for pos in positions_2d_col:
        light_sources_config_2d.append({
            'OptEl_to_world_translation_matrix': pos,
            "OptEl_to_world_rotation_matrix": rotation_matrix_2d
        })
        
    return light_sources_config_2d

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
        self.n1, self.n2, self.n3 = 1.0, 1.51, 1.0
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

    def _find_intersection_newton(self, ray_origin, ray_dir, surface_spline):
        """
        使用牛顿法快速求解光线与样条曲线的交点参数 t。
        直接接收一个 CubicSpline 对象。
        """
        t = np.ones(ray_origin.shape[1]) * 5.0 # 初始猜测
        for _ in range(10): # 牛顿法迭代
            x_intersect = ray_origin[0, :] + t * ray_dir[0, :]
            y_ray = ray_origin[1, :] + t * ray_dir[1, :]
            
            # 直接从样条对象获取 y 和 dy/dx
            y_surface = surface_spline(x_intersect)
            dy_dx = surface_spline(x_intersect, nu=1)
            
            f_t = y_ray - y_surface
            f_prime_t = ray_dir[1, :] - dy_dx * ray_dir[0, :]
            
            f_prime_t[np.abs(f_prime_t) < 1e-9] = 1e-9
            
            delta_t = -f_t / f_prime_t
            t += delta_t
            
            if np.all(np.abs(delta_t) < 1e-7): break
        return t

    def trace_ray(self, p_wcs, n_wcs):
        """
        Traces rays through a 2D lens and returns the points at each stage.
        
        Modified Logic: This version ONLY retains rays that successfully intersect and
        refract through BOTH the lower and upper surfaces.
        """
        # 1. Coordinate Transformation
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, is_vector=True)
        
        # 2. Intersection with the lower surface
        t1 = self._find_intersection_newton(pl0_bcs, n1_bcs, self.down_spline)
        # Filter out invalid intersections immediately
        t1[t1 < 1e-5] = np.nan 
        valid_t1_mask = ~np.isnan(t1)
        if not np.any(valid_t1_mask):
            return [np.empty((2, 0))] * 4
            
        pl0_bcs, n1_bcs, t1 = pl0_bcs[:, valid_t1_mask], n1_bcs[:, valid_t1_mask], t1[valid_t1_mask]
        pl1_bcs = pl0_bcs + t1 * n1_bcs
        
        # 3. First Filtering: Based on boundary of the lower surface
        pl1_filtered, (pl0_filtered, n1_filtered), _ = filter_rays_by_boundary(
            pl1_bcs, (self.bound[0], self.bound[1]), pl0_bcs, n1_bcs
        )
        if pl1_filtered.shape[1] == 0:
            return [np.empty((2, 0))] * 4

        # 4. Refraction at the lower surface
        normal1 = self.down_surface_normal(pl1_filtered[0, :])
        n2_refracted = calculate_refraction_vector(self.n1, n1_filtered, self.n2, normal1)
        
        # 5. Second Filtering: Based on Total Internal Reflection (TIR)
        mask_no_tir1 = ~np.isnan(n2_refracted[0, :])
        pl0_after_tir = pl0_filtered[:, mask_no_tir1]
        pl1_after_tir = pl1_filtered[:, mask_no_tir1]
        n2_after_tir = n2_refracted[:, mask_no_tir1]
        
        if pl1_after_tir.shape[1] == 0:
            return [np.empty((2, 0))] * 4

        # ====================================================================
        # ## 核心修改逻辑 ##
        # ====================================================================

        # 6. Calculate intersection with the upper surface
        t2 = self._find_intersection_newton(pl1_after_tir, n2_after_tir, self.up_spline)
        t2[t2 < 1e-5] = np.nan # Filter out invalid intersections immediately
        pl2_intersections = pl1_after_tir + t2 * n2_after_tir

        # 7. Final Filtering: Filter rays based on upper surface boundary
        #    This single step replaces the previous "classify and process separately" logic.
        pl2_final, (pl0_final, pl1_final, n2_final), mask_hit_up = filter_rays_by_boundary(
            pl2_intersections, (self.bound[0], self.bound[1]),
            pl0_after_tir, pl1_after_tir, n2_after_tir
        )

        # If no rays hit the upper surface, return empty arrays
        if pl2_final.shape[1] == 0:
            return [np.empty((2, 0))] * 4

        # 8. Refraction at the upper surface for the successfully hit rays
        normal2 = self.up_surface_normal(pl2_final[0, :])
        n3_final = calculate_refraction_vector(self.n2, n2_final, self.n3, normal2)
        
        # Additional filter for TIR on the second surface
        mask_no_tir2 = ~np.isnan(n3_final[0, :])
        if not np.any(mask_no_tir2):
            return [np.empty((2,0))] * 4

        # Apply the final TIR filter to all ray path points and vectors
        p0_wcs = self.OptEl_coordinate_to_world_coordinate(pl0_final[:, mask_no_tir2])
        p1_wcs = self.OptEl_coordinate_to_world_coordinate(pl1_final[:, mask_no_tir2])
        p2_wcs = self.OptEl_coordinate_to_world_coordinate(pl2_final[:, mask_no_tir2])
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_final[:, mask_no_tir2], is_vector=True)
        
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

