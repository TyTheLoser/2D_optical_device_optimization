import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from typing import Tuple
from matplotlib.collections import LineCollection

# --- 点集生成函数 ---

def generate_points_on_plane(n, x_range=(-5, 5), y=10):
    """
    在2D空间中，沿一条与X轴平行的直线创建等间距的点。

    :param n: 点的数量。
    :param x_range: 点的x坐标范围 (x_min, x_max)。
    :param y: 所有点的固定y坐标。
    :return: 形状为 (2, n) 的点坐标数组。
    """
    x = np.linspace(x_range[0], x_range[1], n)
    y_coords = np.full_like(x, y)
    points = np.row_stack((x, y_coords))
    return points

def generate_points_on_surface(points_count, equation_params, equation_fun, x_range=(-5, 5)):
    """
    在2D曲线方程上，根据给定的x坐标范围创建点。

    :param points_count: 要创建的点的数量。
    :param equation_params: 曲线方程的参数。
    :param equation_fun: 计算y值的函数，形式为 y = f(params, x)。
    :param x_range: x坐标的范围 (x_min, x_max)。
    :return: 形状为 (2, points_count) 的点坐标数组。
    """
    x = np.linspace(x_range[0], x_range[1], points_count)
    y = equation_fun(equation_params, x)
    points = np.row_stack((x, y))
    return points

# --- 交点计算函数 ---

def calculate_ray_plane_intersections(ray_origins, ray_directions, line_normal, line_point):
    """
    计算2D光线与一条无限长直线的交点。

    :param ray_origins: np.ndarray, 光线起始点，形状 (2, N)。
    :param ray_directions: np.ndarray, 光线方向向量，形状 (2, N)。
    :param line_normal: np.ndarray, 直线的单位法线向量, 形状 (2,)。
    :param line_point: np.ndarray, 直线上的任意一点, 形状 (2,)。
    :return: (intersection_points, mask)
             intersection_points: np.ndarray, 交点的二维坐标, 形状 (2, K)，K是有效交点数。
             mask: np.ndarray, 布尔掩码，标记哪些原始光线产生了有效交点。
    """
    dot_v_n = np.dot(ray_directions.T, line_normal)
    valid_mask = np.abs(dot_v_n) > 1e-6
    if not np.any(valid_mask):
        return np.array([]).reshape(2, 0), np.zeros(ray_origins.shape[1], dtype=bool)

    origins_valid = ray_origins[:, valid_mask]
    dirs_valid = ray_directions[:, valid_mask]
    dot_v_n_valid = dot_v_n[valid_mask]
    
    w = line_point[:, np.newaxis] - origins_valid
    t = np.sum(w * line_normal[:, np.newaxis], axis=0) / dot_v_n_valid
    
    intersection_points = origins_valid + dirs_valid * t
    return intersection_points, valid_mask
    
def calculate_line_surface_intersection_points_by_binary_search(
        line_points, line_vectors, surface_fun, min_range=0, max_range=20, tolerance=1e-6, max_iterations=1000):
    """
    通过二分法求解2D光线与曲线的交点坐标。
    """
    def objective_function(t_value):
        result = line_points + line_vectors * t_value
        # 曲线方程 y_curve = f(x)
        y_surface = surface_fun(x=result[0, :])
        # 目标是找到 y_line(t) - y_surface(x(t)) = 0 的解
        return (result[1, :] - y_surface).reshape((1, -1))

    success_mask, t_values = binary_search(
        func=objective_function,
        min_range=np.ones(line_points.shape[1]) * min_range,
        max_range=np.ones(line_points.shape[1]) * max_range,
        tolerance=tolerance, max_iterations=max_iterations
    )
    intersection_points = line_points + line_vectors * t_values
    return success_mask, intersection_points


def calculate_line_surface_intersection_points(parameters, line_points, line_vectors):
    """
    通过 sympy 库求解2D光线和多项式曲线的交点。
    """
    x, y, t = sp.symbols('x y t')
    # 2D 曲线方程 y = f(x) -> f(x) - y = 0
    curve_equation = sp.Eq(
        parameters[0] + parameters[1] * x + parameters[2] * x**2 + parameters[3] * x**3 - y, 0
    )
    intersections = []
    num_rays = line_points.shape[1]
    for i in range(num_rays):
        point = line_points[:, i]
        direction = line_vectors[:, i]
        
        line_eq1 = sp.Eq(x, direction[0] * t + point[0])
        line_eq2 = sp.Eq(y, direction[1] * t + point[1])
        
        solutions = sp.solve((curve_equation, line_eq1, line_eq2), (x, y, t))

        if isinstance(solutions, dict):
            intersections.append([solutions[x], solutions[y]])
        elif solutions:
            # 筛选出实数解
            real_solutions = [sol for sol in solutions if all(val.is_real for val in sol)]
            if real_solutions:
                intersections.append(real_solutions[0][:2]) # 只取 x, y
            else:
                print(f'\nWarning: Ray {i} has no real solution!')
        
        print(f'\rcalculate process: {((i + 1) / num_rays * 100):.2f}%', end='')
    
    print(f'\rcalculate process: 100.00%')
    return np.array(intersections, dtype=float).T


def calculate_line_plane_intersection_points(line_positions, line_vectors, A, B, C):
    """
    计算2D光线和直线 Ax + By + C = 0 的交点坐标。
    """
    line_normal = np.array([A, B])
    # 直线上的一点，例如如果 B!=0, x=0, y=-C/B
    if B != 0:
        line_point = np.array([0, -C / B])
    elif A != 0:
        line_point = np.array([-C / A, 0])
    else:
        # A=B=0, C=0 is the whole plane, C!=0 is no solution
        return np.full_like(line_positions, np.nan)
        
    intersections, _ = calculate_ray_plane_intersections(line_positions, line_vectors, line_normal, line_point)
    # 此简化函数假设所有光线都有交点
    return intersections


# --- 可视化函数 ---

def visualize_point_cloud(points, new_fig=True, ax=None, **kwargs):
    """
    可视化2D点云数据。
    """
    if new_fig:
        fig, ax = plt.subplots()
    elif ax is None:
        raise ValueError("If new_fig is False, an ax object must be provided.")

    ax.scatter(points[0, :], points[1, :], s=10, **kwargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    return ax

def visualize_plane(parameters, points=100, xmin=-10, xmax=10, new_fig=True, ax=None, **kwargs):
    """
    可视化2D中的一条直线或曲线。
    """
    x = np.linspace(xmin, xmax, points)
    # 假设 'calculate_surface_values' 适配了2D
    y = calculate_surface_values(params=parameters, x=x)

    if new_fig:
        fig, ax = plt.subplots()
    elif ax is None:
        raise ValueError("If new_fig is False, an ax object must be provided.")

    ax.plot(x, y, **kwargs)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    return ax

# --- 向量和几何计算 ---

def line_equation(line_points, line_vectors, t):
    """
    计算2D直线参数方程 P = P0 + t*V 的点坐标。
    """
    return line_points + line_vectors * t

def calculate_direction_vector(start_point, end_points):
    """
    计算从 start_point 出发，指向 end_points 的单位方向向量。
    """
    vectors = end_points - start_point
    return vectors / np.linalg.norm(vectors, axis=0)

def calculate_reflection_vector(incident_direction, normal_direction):
    """
    根据入射光线和法线计算反射光线的2D方向向量。
    """
    incident_norm = incident_direction / np.linalg.norm(incident_direction, axis=0)
    normal_norm = normal_direction / np.linalg.norm(normal_direction, axis=0)
    dot_product = np.sum(incident_norm * normal_norm, axis=0)
    reflection_vector = incident_norm - 2 * dot_product * normal_norm
    return reflection_vector / np.linalg.norm(reflection_vector, axis=0)

def calculate_refraction_vector(n_1, incident_direction, n_2, normal_direction):
    """
    计算2D折射方向向量。
    """
    incident_direction = incident_direction / np.linalg.norm(incident_direction, axis=0)
    normal_direction = normal_direction / np.linalg.norm(normal_direction, axis=0)
    
    # 确保法线和入射方向在对侧
    cos_theta_i = np.sum(incident_direction * normal_direction, axis=0)
    sign = np.sign(cos_theta_i)
    normal_direction = -sign * normal_direction # 调整法线方向
    cos_theta_i = -sign * cos_theta_i # 入射角余弦值应为正

    n_ratio = n_1 / n_2
    radicand = 1 - n_ratio**2 * (1 - cos_theta_i**2)
    
    # 处理全内反射
    total_internal_reflection_mask = radicand < 0
    refraction_vector = np.full_like(incident_direction, np.nan)
    
    valid_mask = ~total_internal_reflection_mask
    if np.any(valid_mask):
        cos_theta_t = np.sqrt(radicand[valid_mask])
        refraction_vector[:, valid_mask] = (n_ratio * incident_direction[:, valid_mask] + 
                                            (cos_theta_t - n_ratio * cos_theta_i[valid_mask]) * normal_direction[:, valid_mask])

    return refraction_vector

def calculate_normal_vector(n_1, incident_direction, n_2, refracted_direction):
    """
    根据入射和折射向量反推2D法向量。
    """
    normal_vector = n_2 * refracted_direction - n_1 * incident_direction
    return -normal_vector / np.linalg.norm(normal_vector, axis=0)

def calculate_surface_values(params, x):
    """
    根据一维多项式参数，计算曲线在x点处的y值。
    """
    # 假设参数对应: c, b, a, d -> y = a*x^2 + b*x + c + d*x^3
    return (params[0] + params[1] * x + params[2] * x**2 + params[3] * x**3)
    

def calculate_surface_normal_vectors(parameters, x, standard_normal_direction):
    """
    计算2D曲线上各点处的法向量 y = f(x) -> g(x,y) = y-f(x)=0, Normal=grad(g)=(-f', 1)。
    """
    # f'(x) = b + 2ax + 3dx^2
    df_dx = parameters[1] + 2 * parameters[2] * x + 3 * parameters[3] * x**2
    
    normal_vectors = np.row_stack((-df_dx, np.ones_like(x)))
    normal_vectors_normalized = normal_vectors / np.linalg.norm(normal_vectors, axis=0)
    
    # 确保法线方向与标准方向一致
    dot_product = np.sum(normal_vectors_normalized * standard_normal_direction, axis=0)
    # 如果点积为负，翻转法线方向
    normal_vectors_normalized[:, dot_product < 0] *= -1
    
    return normal_vectors_normalized

def calculate_surface_value_and_normal_vectors(parameters, data):
    """
    计算2D曲线上各点的值（y坐标）和法向量。
    """
    x = data[0]
    y_values = calculate_surface_values(params=parameters, x=x)
    
    data_new = data.copy()
    data_new[1] = y_values
    
    normal_vectors = calculate_surface_normal_vectors(
        parameters=parameters, x=x, standard_normal_direction=np.array([0, -1]).reshape((2, 1))
    )
    return data_new, normal_vectors

# --- 差异与距离计算 ---

def vector_difference_cross(vector1, vector2):
    """
    判断两个2D向量的方向一致性，使用2D伪叉乘。
    结果越接近零，方向越一致（或相反）。
    """
    # 2D 叉乘: x1*y2 - x2*y1
    pseudo_cross_product = vector1[0, :] * vector2[1, :] - vector1[1, :] * vector2[0, :]
    return np.abs(pseudo_cross_product)

def calculate_distance_difference(points1, points2):
    """
    计算两组2D点集之间对应点的欧几里得距离。
    """
    return np.linalg.norm(points1 - points2, axis=0)

def point_distance_to_lines(point, line_positions, line_vectors):
    """
    计算一个2D点到多条空间直线的距离。
    """
    vec_to_point = point.reshape((2, 1)) - line_positions
    
    # 2D 叉乘: | (P-O) x V | / |V|
    cross_product_mag = np.abs(vec_to_point[0, :] * line_vectors[1, :] - vec_to_point[1, :] * line_vectors[0, :])
    vector_mag = np.linalg.norm(line_vectors, axis=0)
    
    # 避免除以零
    distances = np.divide(cross_product_mag, vector_mag, out=np.zeros_like(vector_mag), where=vector_mag!=0)
    return distances

# --- 求解器与过滤器 ---

def binary_search(func, min_range, max_range, retry_times=8, tolerance=1e-6, max_iterations=100, parameters=None):
    """
    利用二分法寻找对应的解，此函数逻辑与维度无关，无需修改。
    """
    current_max_range = np.copy(max_range)
    num_total_rays = len(min_range)
    
    for _ in range(retry_times + 1):
        val_min = func(min_range)
        val_max = func(current_max_range)
        good_mask = (val_min * val_max <= 0).flatten()
        if np.all(good_mask):
            break
        bad_mask = ~good_mask
        current_max_range[bad_mask] += 10

    final_good_mask = (func(min_range) * func(current_max_range) <= 0).flatten()
    results = np.full(num_total_rays, np.nan, dtype=float)
    
    if not np.any(final_good_mask):
        print("Error: No valid brackets found for any item after all retries.")
        return final_good_mask, results

    min_r = np.copy(min_range)
    max_r = np.copy(current_max_range)

    for i in range(max_iterations):
        center_r = (min_r + max_r) / 2
        
        if np.all(np.abs(max_r[final_good_mask] - min_r[final_good_mask]) / 2 < tolerance):
            break

        f_center = func(center_r).flatten()
        f_min = func(min_r).flatten()
        
        change_to_max = f_center * f_min < 0

        update_max_mask = change_to_max & final_good_mask
        update_min_mask = (~change_to_max) & final_good_mask
        
        max_r[update_max_mask] = center_r[update_max_mask]
        min_r[update_min_mask] = center_r[update_min_mask]
        
    final_t_values = (min_r + max_r) / 2
    results[final_good_mask] = final_t_values[final_good_mask]
    
    success_mask = ~np.isnan(results)
    return success_mask, results


def filter_rays_by_boundary(
    points_to_check: np.ndarray,
    boundary_x: tuple,
    boundary_y: tuple,
    *arrays_to_filter: np.ndarray
) -> tuple:
    """
    根据一个矩形边界来筛选2D光线数据。此函数逻辑与3D版本兼容，无需修改。
    """
    if points_to_check.shape[1] == 0:
        return np.array([]).reshape(2, 0), tuple(arr.reshape(arr.shape[0], 0) for arr in arrays_to_filter), np.array([], dtype=bool)

    mask = (points_to_check[0, :] >= boundary_x[0]) & (points_to_check[0, :] <= boundary_x[1]) & \
           (points_to_check[1, :] >= boundary_y[0]) & (points_to_check[1, :] <= boundary_y[1])

    points_inside = points_to_check[:, mask]
    filtered_arrays = tuple(arr[:, mask] for arr in arrays_to_filter)
    return points_inside, filtered_arrays, mask
def plot_2d_rays(ax, start_points, end_points, num_to_plot=100000, **kwargs):
    """高效绘制2D光线路径。"""
    num_rays = min(start_points.shape[1], end_points.shape[1])
    if num_rays == 0: return
    indices = np.random.choice(num_rays, min(num_rays, num_to_plot), replace=False)
    segments = np.array([start_points[:, indices].T, end_points[:, indices].T]).transpose((1, 0, 2))
    ax.add_collection(LineCollection(segments, **kwargs))
def visualize_scene(ax, title, source_list, device, evaluator, rays, hit_points):
    """
    可视化整个光学场景，支持多个光源。

    Args:
        ax (matplotlib.axes.Axes): 绘图轴。
        title (str): 图像标题。
        source_list (list): 包含一个或多个光源对象的列表。
        device (object): 光学器件对象。
        evaluator (object): 评估平面对象。
        rays (tuple): 包含光线路径数据的元组 (p0, p1, p2, n3)。
        hit_points (numpy.ndarray): 在评估平面上的命中点坐标。
    """
    p0, p1, p2, n3 = rays
    ax.set_facecolor('black')
    ax.set_title(title, fontsize=16)
    plot_2d_rays(ax, p0, p1, colors='orange', linewidths=0.4, alpha=0.6)
    plot_2d_rays(ax, p1, p2, colors='deepskyblue', linewidths=0.4, alpha=0.7)
    p3_extended = p2 + n3 * 40
    plot_2d_rays(ax, p2, p3_extended, colors='lime', linewidths=0.5, alpha=0.8)
    # 遍历光源列表并绘制每个光源
    for source in source_list:
        source.plot_element_2d(ax, zorder=10)

    device.plot_element_2d(ax, color='cyan', label='Lens', zorder=5)
    evaluator.plot_element_2d(ax, color='magenta', linestyle='--', linewidth=3, zorder=10)

    

    if hit_points.shape[1] > 0:
        ax.scatter(hit_points[0, :], hit_points[1, :], s=5, c='red', alpha=0.8, label='HitPoints', zorder=11)

    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.legend(); ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xlim(-5, 20); ax.set_ylim(-5, 45)