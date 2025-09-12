import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


def generate_points_on_plane(n, x_range=(-5, 5), y_range=(-5, 5), z=100):
    """
    在对应的 z 平面上创建点
    :param n:
    :param x_range:
    :param y_range:
    :param z:
    :return:
    """
    x = np.linspace(x_range[0]+3, x_range[1]-3, num=int(np.sqrt(n)))
    y = np.linspace(y_range[0], y_range[1], num=int(np.sqrt(n)))
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x) * z

    points = np.row_stack((x.flatten(), y.flatten(), z.flatten()))
    return points[:n]


def generate_points_on_surface(points_count, equation_params, equation_fun, x_range=(-5, 5), y_range=(-5, 5)):
    """
    在对应的曲面方程上对应的 x 和 y 点对应的交点位置创建光线
    :param equation_fun:
    :param equation_params:
    :param points_count:
    :param x_range:
    :param y_range:
    :param z:
    :return:
    """
    x = np.linspace(x_range[0], x_range[1], num=int(np.sqrt(points_count)))
    y = np.linspace(y_range[0], y_range[1], num=int(np.sqrt(points_count)))
    x, y = np.meshgrid(x, y)
    x_flatten = x.flatten()
    y_flatten = y.flatten()
    z = equation_fun(equation_params, x_flatten, y_flatten)

    points = np.row_stack((x_flatten, y_flatten, z))
    return points
def calculate_ray_plane_intersections(ray_origins, ray_directions, plane_normal, plane_point):
    """
    计算光线与一个无限大平面的交点。

    此函数采用标准线面交点公式，具有很强的通用性。
    平面由法线向量(plane_normal)和平面上的一点(plane_point)定义。

    :param ray_origins: np.ndarray, 光线起始点，形状 (3, N)。
    :param ray_directions: np.ndarray, 光线方向向量，形状 (3, N)。
    :param plane_normal: np.ndarray, 平面的单位法线向量, 形状 (3,)。
    :param plane_point: np.ndarray, 平面上的任意一点, 形状 (3,)。
    :return: (intersection_points, mask)
             intersection_points: np.ndarray, 交点的三维坐标, 形状 (3, K)，K是有效交点数。
             mask: np.ndarray, 布尔掩码，标记哪些原始光线产生了有效交点。
    """
    # V.N, 如果点积为0，则光线与平面平行
    dot_v_n = np.dot(ray_directions.T, plane_normal)

    # 找到不平行的光线 (dot积的绝对值大于一个很小的数)
    valid_mask = np.abs(dot_v_n) > 1e-6
    if not np.any(valid_mask):
        return np.array([]).reshape(3, 0), np.zeros(ray_origins.shape[1], dtype=bool)

    # 筛选出有效的光线进行计算
    origins_valid = ray_origins[:, valid_mask]
    dirs_valid = ray_directions[:, valid_mask]
    dot_v_n_valid = dot_v_n[valid_mask]
    
    # 经典线面交点公式中的参数 t
    # t = ((P_plane - P0) . N) / (V . N)
    w = plane_point[:, np.newaxis] - origins_valid
    t = np.sum(w * plane_normal[:, np.newaxis], axis=0) / dot_v_n_valid
    
    # 交点 P = P0 + t * V
    intersection_points = origins_valid + dirs_valid * t
    
    return intersection_points, valid_mask

def visualize_point_cloud(points, new_fig=True, ax=None):
    """
    可视化点云数据
    :param points:
    :param new_fig:
    :param ax:
    :return:
    """
    # 如果 new_fig 为 True，则创建新的图形窗口
    if new_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    elif ax is None:
        raise ValueError("If new_fig is False, an ax object must be provided.")

    ax.scatter(points[0, :], points[1, :], points[2, :], s=5, c='b', marker='o')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def visualize_plane(parameters, points=100, xmin=-10, xmax=10, ymin=-10, ymax=10,
                    new_fig=True, ax=None):
    """
    可视化平面
    :param parameters:
    :param points:
    :param xmin:
    :param xmax:
    :param ymin:
    :param ymax:
    :param new_fig:
    :param ax:
    :return:
    """
    # 生成平面上的均匀分布的点
    x = np.linspace(xmin, xmax, points)
    y = np.linspace(ymin, ymax, points)
    x, y = np.meshgrid(x, y)
    z = calculate_surface_values(params=parameters, x=x, y=y)

    # 如果 new_fig 为 True，则创建新的图形窗口
    if new_fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    elif ax is None:
        raise ValueError("If new_fig is False, an ax object must be provided.")

    # 绘制平面
    ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color='b')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax

def line_equation(line_points, line_vectors, t):
    """
    t 的 shape 和 line_points 的 shape 一样
    :param t:
    :return:
    """
    result = line_points + line_vectors * t
    return result


def calculate_line_surface_intersection_points_by_binary_search(
        line_points, line_vectors, surface_fun, min_range=0, max_range=20, tolerance=1e-6, max_iterations=1000):
    """
    通过二分法求解对应的交点坐标。
    此版本确保始终返回与输入大小相同的数组，失败处用NaN填充。
    """
    def objective_function(t_value):
        # (objective_function 保持不变)
        result = line_points + line_vectors * t_value
        z_surface = surface_fun(x=result[0, :], y=result[1, :])
        return (result[2, :] - z_surface).reshape((1, -1))

    # --- 关键修改 ---
    
    # 1. 调用我们之前修正好的、支持部分失败的 binary_search
    success_mask, t_values = binary_search(
        func=objective_function, 
        min_range=np.ones(line_points.shape[1]) * min_range,
        max_range=np.ones(line_points.shape[1]) * max_range,
        tolerance=tolerance, max_iterations=max_iterations
    )

    # 2. 直接使用包含 NaN 的 t_values 进行计算
    #    NumPy 会自动将 NaN 传播到计算结果中，这正是我们想要的。
    #    line_points (3, 500) + line_vectors (3, 500) * t_values (500,)
    #    这里的 t_values 长度与 line_points/line_vectors 的列数相同。
    intersection_points = line_points + line_vectors * t_values

    # 3. 返回的 intersection_points 数组形状将与 line_points 完全相同
    #    例如 (3, 500)，失败的光线对应的列将是 [nan, nan, nan]。
    return success_mask, intersection_points


def calculate_line_surface_intersection_points(parameters, line_points, line_vectors):
    """
    通过 sp 库来求解对应的线和曲面的交点
    :param parameters:
    :param line_points:
    :param line_vectors:
    :return:
    """
    x, y, z, t = sp.symbols('x y z t')
    surface_equation = sp.Eq((parameters[0] * x + parameters[1] * y + parameters[2]
                              + parameters[3] * x ** 2 + parameters[4] * y ** 2 + parameters[5] * x * y
                              + parameters[6] * x ** 3 + parameters[7] * x ** 2 * y + parameters[8] * x * y ** 2 +
                              parameters[9] * y ** 3
                              - z), 0)

    intersections = []

    i = 0
    all_len = line_points.shape[1]
    for point, direction in zip(line_points.T, line_vectors.T):
        line_equation1 = sp.Eq(x, direction[0] * t + point[0])
        line_equation2 = sp.Eq(y, direction[1] * t + point[1])
        line_equation3 = sp.Eq(z, direction[2] * t + point[2])
        # 解方程组
        solutions = sp.solve((surface_equation, line_equation1, line_equation2, line_equation3), (x, y, z, t))
        # if len(solutions) != 1:
        #     result1 = solutions[0]
        #     intersections.append([result1[0], result1[1], result1[2]])
        # else:
        #     intersections.append([solutions[x], solutions[y], solutions[z]])
        if type(solutions) == dict:
            intersections.append([solutions[x], solutions[y], solutions[z]])
        else:
            is_there_a_valid_solution = False
            for i in range(len(solutions)):
                # 如果x和y都不是复数才可以
                if type(solutions[i][0]) != sp.core.add.Add and type(solutions[i][1]) != sp.core.add.Add:
                    solutions_0 = solutions[i]
                    intersections.append([solutions_0[0], solutions_0[1], solutions_0[2]])
                    is_there_a_valid_solution = True
                    break
            if is_there_a_valid_solution is False:
                print('\nno valid solution!')
        i += 1
        print('\rcalculate process: {:.2f}%'.format(i / all_len * 100), end='')

    print('\rcalculate process: {:.2f}%'.format(100.00))
    return np.array(intersections, dtype=float).T


def calculate_line_plane_intersection_points(line_positions, line_vectors, A, B, C, D):
    """
    计算光线和平面的交点坐标
    :param line_positions:
    :param line_vectors:
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """
    # 平面的系数
    plane_vector = np.array([A, B, C]).reshape((3, 1))  # 计算平面的法线向量
    plane_vector = plane_vector / np.linalg.norm(plane_vector)
    plane_vector = np.repeat(plane_vector, repeats=line_vectors.shape[1], axis=1)
    # vpt = np.dot(V.T, plane_vector)
    vpt = np.sum(np.multiply(line_vectors, plane_vector), axis=0)
    # 平面上的一点
    plane_point = np.zeros_like(line_positions)
    if A != 0:
        plane_point[0, :] = -D / A
    elif B != 0:
        plane_point[1, :] = -D / B
    elif C != 0:
        plane_point[2, :] = -D / C

    # 判断直线是否与平面平行
    parallel_mask = vpt == 0
    return_result = np.zeros_like(line_positions)

    # 计算交点
    # t = np.dot((plane_point - P).T, plane_vector / vpt)
    # return_result = (P.T + V.T * t).T
    t = np.sum(np.multiply(plane_point - line_positions, plane_vector / vpt), axis=0)
    return_result = (line_positions.T + np.multiply(line_vectors.T, np.repeat(t.reshape((-1, 1)), 3, axis=1))).T

    return return_result


def calculate_direction_vector(start_point, points_on_plane):
    """
    计算从 start_point 出发，指向 points_on_plane 的单位方向向量
    :param start_point:
    :param points_on_plane:
    :return:
    """
    # 计算起始点到平面上所有点的向量
    vectors = points_on_plane - np.repeat(start_point, repeats=points_on_plane.shape[1], axis=1)

    # 计算方向向量
    direction_vectors = vectors / np.linalg.norm(vectors, axis=0)

    return direction_vectors

def calculate_reflection_vector(incident_direction, normal_direction):
    """
    根据入射光线和法线计算反射光线的方向向量。
    该函数支持向量化操作，可同时计算多条光线。

    :param incident_direction: 入射光线的方向向量 (3xN ndarray)，向量应指向表面。
    :param normal_direction: 反射点处的表面法向量 (3xN ndarray)，法向量应从表面向外指出。
    :return: 反射光线的方向向量 (3xN ndarray)。
    """
    # 1. 归一化输入向量，确保它们是单位向量
    incident_norm = incident_direction / np.linalg.norm(incident_direction, axis=0)
    normal_norm = normal_direction / np.linalg.norm(normal_direction, axis=0)

    # 2. 计算入射向量和法向量的点积
    # 对于 (3, N) 的数组, 逐元素相乘后再按列求和 (axis=0) 即可得到 N 个点积结果。
    # np.sum(A * B, axis=0) 是向量化计算点积的常用方法。
    dot_product = np.sum(np.multiply(incident_norm, normal_norm), axis=0)

    # 3. 应用反射向量的通用公式: R = I - 2 * (I · N) * N
    # I 是归一化的入射向量 (incident_norm)
    # N 是归一化的法向量 (normal_norm)
    # (I · N) 是它们的点积 (dot_product)
    # NumPy的广播机制会自动将 (1, N) 的 dot_product 数组正确地应用到 (3, N) 的 normal_norm 数组上。
    reflection_vector = incident_norm - 2 * dot_product * normal_norm

    # 4. (可选但推荐) 再次归一化最终结果以消除任何潜在的浮点计算误差
    reflection_vector_norm = reflection_vector / np.linalg.norm(reflection_vector, axis=0)

    return reflection_vector_norm
def calculate_refraction_vector(n_1, incident_direction, n_2, normal_direction):
    """
    计算折射方向向量
    :param n_1: 入射光线所在平面的折射率
    :param incident_direction: 入射光线的方向向量(向量区分方向正负)
    :param n_2: 折射光线所在平面的折射率
    :param normal_direction: 折射点处法向量(入射光线空间方向的平面向量)
    :return:
    """
    # 归一化
    incident_direction = incident_direction / np.linalg.norm(incident_direction, axis=0)
    normal_direction = normal_direction / np.linalg.norm(normal_direction, axis=0)
    # 计算入射角
    cos_theta_i = np.sum(np.multiply(incident_direction, normal_direction), axis=0)
    # sin_theta_i = np.sqrt(1 - cos_theta_i ** 2)
    # # 根据折射定律计算折射角
    # sin_theta_r = (n_1 / n_2) * sin_theta_i
    # cos_theta_r = np.sqrt(1 - sin_theta_r ** 2)

    # 针对可能出现负数的情景处理
    tmp = n_2 ** 2 - n_1 ** 2 * (1 - (cos_theta_i ** 2))
    if np.any(tmp < 0):
        raise ValueError('根号不能为负数')
    else:
        p = np.sqrt(tmp) - n_1 * cos_theta_i

    # # 针对负数不处理
    # p = np.sqrt(n_2 ** 2 - n_1 ** 2 * (1 - (cos_theta_i ** 2))) - n_1 * cos_theta_i

    # 计算折射方向向量
    refraction_vector = (n_1 * incident_direction + np.repeat(p.reshape((1, -1)), repeats=3,
                                                              axis=0) * normal_direction) / n_2

    # 归一化法线向量
    refraction_vector = refraction_vector / np.linalg.norm(refraction_vector, axis=0)

    return refraction_vector


def calculate_normal_vector(n_1, incident_direction, n_2, refracted_direction):
    """
    返回的法线方向向量和对应的入射光线的方向向量处于同一个方向
    :param n_1:
    :param incident_direction:
    :param n_2:
    :param refracted_direction:
    :return:
    """
    # 计算差向量 (A_0' - nA_0) = p*N
    normal_vector = n_2 * refracted_direction - n_1 * incident_direction

    # 归一化法线向量
    normal_vector = -normal_vector / np.linalg.norm(normal_vector, axis=0)

    return normal_vector


# def calculate_surface_normal_vector(differential_equation, standard_direction, x, y):
#     """
#     利用偏微分方程的方法计算平面的法向量，参数最好设置为一个函数的地址，这样子法向量不用手动再求解，便于迭代更新
#     :param surface_equation: 待求解法向量的平面方程
#     :param standard_direction: 法向量的标准方向，用于判断法向量的正反方向
#     :return:
#     """
#     # 法向量
#     normal_vector = differential_equation(x, y)
#
#     # ----------------------------------------------------------------
#     # 计算法向量与标准方向的点积
#     dot_product = np.dot(normal_vector.T, standard_direction)
#     # 如果点积为负数，说明法向量与标准方向相反，需要调整方向
#     if dot_product < 0:
#         adjusted_normal = -normal_vector
#     else:
#         adjusted_normal = normal_vector
#
#     return adjusted_normal

def calculate_surface_value_and_normal_vectors(parameters, data):
    """
    作为对应的模型函数，需要返回的内容不仅有对应的拟合曲面的坐标，还要有不同位置的曲面对应的法向量
    :param parameters:
    :param data:
    :return:
    """
    x = data[0]
    y = data[1]

    # 双变量的模型，这里简单地以二次函数为例
    # surface_values = (
    #         parameters[0] * x + parameters[1] * y + parameters[2]  # 原始的方程
    #         + parameters[3] * x ** 2 + parameters[4] * y ** 2 + parameters[5] * x * y  # 二次项
    #         + parameters[6] * x ** 3 + parameters[7] * x ** 2 * y + parameters[8] * x * y ** 2 + parameters[9] * y ** 3
    #     # 三次项
    # )
    surface_values = calculate_surface_values(params=parameters, x=x, y=y)

    # 返回的仍然是一个三维向量
    data_new = data.copy()
    data_new[2] = surface_values

    # 计算法向量（这里仅是示例，具体法向量的计算需要根据具体模型而定）
    # normal_vectors = np.array([
    #     -(2 * parameters[3] * x + parameters[5] * y + 3 * parameters[6] * x ** 2
    #       + 2 * parameters[7] * x * y + parameters[8] * y ** 2),
    #     -(parameters[0] + parameters[5] * x + parameters[7] * x ** 2 + 2 * parameters[8] * x * y
    #       + 3 * parameters[9] * y ** 2),
    #     np.ones_like(x)
    # ])
    normal_vectors = calculate_surface_normal_vectors(
        parameters=parameters, x=x, y=y, standard_normal_direction=np.array([0, 0, -1]).reshape((3, 1)))

    return data_new, normal_vectors


def vector_difference_cross(vector1, vector2):
    """
   判断两个向量的方向一致性
   :param vector1: 第一个向量，3*n的numpy数组
   :param vector2: 第二个向量，3*n的numpy数组
   :return: 方向一致性的程度，值越接近零表示方向越一致
   """
    cross_product = np.cross(vector1, vector2, axis=0)  # 计算叉乘
    consistency_measure = np.linalg.norm(cross_product, axis=0)  # 计算叉乘结果的模

    return consistency_measure


def calculate_distance_difference(points1, points2):
    """
    计算不同点之间的距离差异
    :param points1: 第一个点集，3*n的numpy数组
    :param points2: 第二个点集，3*n的numpy数组
    :return: 距离差异的平均值
    """
    distance_difference = np.linalg.norm(points1 - points2, axis=0)  # 计算欧几里得距离
    # average_distance_difference = np.mean(distance_difference)  # 计算平均值

    return distance_difference


def calculate_surface_values(params, x, y):
    """
    给定对应去买呢方程的参数，然后计算得到对应位置的 z 值
    :param params:
    :param x:
    :param y:
    :return:
    """
    return (params[0] * x
            + params[1] * y
            + params[2]  # 原始的方程
            + params[3] * x ** 2
            + params[4] * y ** 2
            + params[5] * x * y  # 二次项
            + params[6] * x ** 3
            + params[7] * x ** 2 * y
            + params[8] * x * y ** 2
            + params[9] * y ** 3  # 三次项
            )


def calculate_surface_normal_vectors(parameters, x, y, standard_normal_direction):
    """
    给定对应曲面方程的参数，然后计算获取到对应位置的法向量
    :param parameters:
    :param x:
    :param y:
    :param standard_normal_direction:
    :return:
    """
    # 计算法向量（这里仅是示例，具体法向量的计算需要根据具体模型而定）
    normal_vector = np.array([
        -(parameters[0]
          + 2 * parameters[3] * x
          + parameters[5] * y
          + 3 * parameters[6] * x ** 2 + 2 * parameters[7] * x * y
          + parameters[8] * y ** 2),
        -(parameters[1]
          + 2 * parameters[4] * y
          + parameters[5] * x
          + parameters[7] * x ** 2
          + 2 * parameters[8] * x * y + 3 * parameters[9] * y ** 2),
        np.ones_like(x)
    ])
    # 归一化
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector, axis=0)

    # ----------------------------------------------------------------
    # 计算法向量与标准方向的点积
    dot_product = np.dot(normal_vector_normalized.T, standard_normal_direction)
    # 如果点积为负数，说明法向量与标准方向相反，需要调整方向
    dot_product = np.repeat(dot_product.T, repeats=3, axis=0)
    adjusted_normal = np.multiply(dot_product, normal_vector_normalized)

    return adjusted_normal


def point_distance_to_lines(point, line_positions, line_vectors):
    """
    计算一个点到对应的空间直线的距离
    :param point: 点的坐标
    :param line_positions: 空间直线的起点
    :param V1: 空间直线的方向向量
    :return: 返回的是点和每一条直线的最短距离
    """
    vec1 = np.repeat(point.reshape((3, 1)), repeats=line_positions.shape[1], axis=1) - line_positions

    distances = (np.linalg.norm(np.cross(vec1, line_vectors, axis=0), axis=0)
                 / np.linalg.norm(line_vectors, axis=0))

    return distances



def binary_search(func, min_range, max_range, retry_times=8, tolerance=1e-6, max_iterations=100, parameters=None):
    """
    利用二分法寻找对应的解，支持对部分失败的项进行重试和标记。
    此版本修正了因处理数组子集而导致的广播错误。
    """
    current_max_range = np.copy(max_range)
    num_total_rays = len(min_range) # 获取总光线数
    
    # --- 1. 括号寻找阶段 (逻辑不变) ---
    for _ in range(retry_times + 1):
        val_min = func(min_range)
        val_max = func(current_max_range)
        good_mask = (val_min * val_max <= 0).flatten()
        if np.all(good_mask):
            break
        bad_mask = ~good_mask
        current_max_range[bad_mask] += 10

    # --- 2. 最终处理阶段 (逻辑不变) ---
    final_good_mask = (func(min_range) * func(current_max_range) <= 0).flatten()
    results = np.full(num_total_rays, np.nan, dtype=float)
    
    if not np.any(final_good_mask):
        print("Error: No valid brackets found for any item after all retries.")
        return final_good_mask, results

    # --- 3. 二分法核心计算 (关键修改) ---
    
    # *** 不再对 min_range 和 max_range 进行切片 ***
    # *** 始终保持它们为完整长度 (N,) ***
    
    # 只需要复制一份用于迭代，以避免修改原始输入
    min_r = np.copy(min_range)
    max_r = np.copy(current_max_range)

    for i in range(max_iterations):
        # 计算所有光线的中间点，center_r 的长度将是 N
        center_r = (min_r + max_r) / 2
        
        # 检查收敛时，只关心那些我们正在处理的“好”光线
        if np.all(np.abs(max_r[final_good_mask] - min_r[final_good_mask]) / 2 < tolerance):
            break

        # func 的输入 center_r 长度为 N，与 line_vectors 匹配，不会再报错
        f_center = func(center_r).flatten()
        f_min = func(min_r).flatten()
        
        change_to_max = f_center * f_min < 0

        # *** 关键：只更新那些在 final_good_mask 中为 True 的光线的边界 ***
        update_max_mask = change_to_max & final_good_mask
        update_min_mask = (~change_to_max) & final_good_mask
        
        max_r[update_max_mask] = center_r[update_max_mask]
        min_r[update_min_mask] = center_r[update_min_mask]
        
    # --- 4. 返回最终结果 ---
    final_t_values = (min_r + max_r) / 2
    # 只将有效光线的结果写入最终的 results 数组
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
    根据一个矩形边界来筛选光线数据。

    函数会检查 `points_to_check` 中的每个点的 (x, y) 坐标是否在指定的边界内，
    然后返回所有在边界内的点，以及在 `arrays_to_filter` 中传入的、相应位置的向量或其他数据。

    :param points_to_check: np.ndarray, 需要进行边界检查的三维点坐标数组, 形状为 (3, N)。
    :param boundary_x: tuple, X方向的边界范围 (x_min, x_max)。
    :param boundary_y: tuple, Y方向的边界范围 (y_min, y_max)。
    :param arrays_to_filter: 任意数量的、需要与 `points_to_check` 同步筛选的数组。
                             它们的第二个维度长度必须也为 N。
    :return: tuple, 返回一个元组，包含：
             - points_inside (np.ndarray): 通过筛选的点的坐标。
             - filtered_arrays (tuple): 一个包含所有被筛选后的其他数组的元组。
             - mask (np.ndarray): 用于筛选的布尔掩码，长度为 N。
    """
    # 首先处理特殊情况：如果没有输入点，则直接返回空值
    if points_to_check.shape[1] == 0:
        return np.array([]).reshape(3, 0), tuple(arr.reshape(arr.shape[0], 0) for arr in arrays_to_filter), np.array([], dtype=bool)

    # 创建布尔掩码，检查 x 和 y 坐标是否在定义的边界内
    mask = (points_to_check[0, :] >= boundary_x[0]) & (points_to_check[0, :] <= boundary_x[1]) & \
           (points_to_check[1, :] >= boundary_y[0]) & (points_to_check[1, :] <= boundary_y[1])

    # 使用掩码筛选主数组（points_to_check）
    points_inside = points_to_check[:, mask]

    # 使用相同的掩码筛选所有其他传入的数组
    filtered_arrays = tuple(arr[:, mask] for arr in arrays_to_filter)

    return points_inside, filtered_arrays, mask


