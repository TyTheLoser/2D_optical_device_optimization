import json
import math
from abc import abstractmethod
from enum import Enum
from stl import mesh
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

from utils.utils_free_form_prism_design import calculate_refraction_vector, binary_search,calculate_reflection_vector,\
    calculate_line_surface_intersection_points_by_binary_search,calculate_ray_plane_intersections,filter_rays_by_boundary
from scipy.spatial.transform import Rotation as R

def is_pure_rotation(matrix):
    """
    检查矩阵是否为纯旋转矩阵（行列式为1）。
    """
    return np.isclose(np.linalg.det(matrix), 1.0)


def decompose_reflection(matrix):
    """
    将矩阵分解为纯旋转矩阵和反射矩阵的乘积。
    返回纯旋转矩阵和反射矩阵。
    """
    if is_pure_rotation(matrix):
        return matrix, np.eye(3)  # 如果是纯旋转，反射矩阵为单位矩阵

    # 如果包含反射，提取反射部分
    reflection_matrix = np.diag([1, 1, -1])  # 假设反射发生在Z轴
    rotation_matrix = matrix @ reflection_matrix.T  # 去除反射部分
    return rotation_matrix, reflection_matrix


def normalize_rotation_vector(rotvec):
    """
    规范化旋转向量，确保旋转角度 <= π。
    """
    theta = np.linalg.norm(rotvec)  # 计算旋转角度
    if theta > np.pi:
        rotvec = -rotvec  # 取反旋转轴
        theta = 2 * np.pi - theta  # 调整旋转角度
    return rotvec


def matrix_to_rotvec(matrix):
    """
    将任意3x3矩阵转换为旋转向量。
    如果是纯旋转矩阵，直接转换；
    如果包含反射，先分解为旋转和反射部分，再转换旋转部分。
    """
    if is_pure_rotation(matrix):
        # 纯旋转矩阵
        r = R.from_matrix(matrix)
        rotvec = r.as_rotvec()
        return normalize_rotation_vector(rotvec), None  # 无反射矩阵
    else:
        # 包含反射的矩阵
        rotation_matrix, reflection_matrix = decompose_reflection(matrix)
        r = R.from_matrix(rotation_matrix)
        rotvec = r.as_rotvec()
        return normalize_rotation_vector(rotvec), reflection_matrix


def rotvec_to_matrix(rotvec, reflection_matrix=None):
    """
    将旋转向量（和可选的反射矩阵）转换回原始矩阵。
    """
    r = R.from_rotvec(rotvec)
    rotation_matrix = r.as_matrix()

    if reflection_matrix is not None:
        # 包含反射的矩阵
        return rotation_matrix @ reflection_matrix
    else:
        # 纯旋转矩阵
        return rotation_matrix


class PrismSide(Enum):
    side_left = 0
    side_right = 1


class TransformOrder(Enum):
    """
    坐标变换方法，先平移后旋转还是先旋转后平移
    """
    TRANSLATE_ROTATE = 1  # 先平移后旋转
    ROTATE_TRANSLATE = 2  # 先旋转后平移


def BFP_calculate_line_surface_intersection_points_by_binary_search(
        line_points, line_vectors, surface_fun, min_range=0, max_range=20, tolerance=1e-6, max_iterations=1000):
    """
    通过二分法求解对应的交点坐标，希望速度和原来的速度相比不会太慢
    :param line_points:
    :param line_vectors:
    :param surface_fun:
    :param line_fun:
    :param min_range:
    :param max_range:
    :param tolerance:
    :param max_iterations:
    :return:
    """

    def objective_function(t_value):
        """
        二分法寻找参数的回调函数
        :param t_value:
        :return: 直线的值和曲面的值的差值
        """
        result = line_points + line_vectors * t_value
        z_surface = surface_fun(x=result[0, :], y=result[1, :])
        return (result[2, :] - z_surface).reshape((1, -1))

    # 将对应的搜寻范围扩大一点，防止无解的情况
    # min_range = -100
    # max_range = 100

    # 神来之笔：
    # 你知道我对你不仅仅是喜欢，你眼里却没有我想要的答案！
    line_vectors = line_vectors / np.linalg.norm(line_vectors, axis=0)

    ret, t_value = binary_search(
        func=objective_function, min_range=np.ones(line_points.shape[1]) * min_range,
        max_range=np.ones(line_points.shape[1]) * max_range,
        tolerance=tolerance, max_iterations=max_iterations, parameters=(line_points, line_vectors)
    )
    if ret:
        intersection_points = line_points + line_vectors * t_value
    else:
        intersection_points = np.zeros_like(line_points)
        print('no valid solution!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')

    return ret, intersection_points


def calculate_points_total_length(points):
    """
    计算给定空间点的所有线段的总长度
    参数：
    points (numpy.ndarray): 一个3xN的NumPy数组，包含N个空间点的坐标
    返回值：
    float: 所有线段的总长度
    """
    # 获取点的数量
    n = points.shape[1]
    # 初始化总长度为0
    total_length = 0
    # 计算所有线段的总长度
    for i in range(n - 1):
        # 计算两点之间的距离
        distance = np.linalg.norm(points[:, i + 1] - points[:, i])
        total_length += distance

    return total_length


def plot_3d_line(points):
    """
    绘制光线路径
    :param points:
    :return:
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    z = [point[2] for point in points]
    for i in range(len(points) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], 'b')
    ax.scatter(x, y, z, c='r', marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.axis("equal")
    plt.show()


def angle_between_vectors(v1, v2, deg_result=True):
    """
    返回的角度单位为度
    :param v1:
    :param v2:
    :param deg_result:
    :return:
    """
    norm_v1 = v1 / np.linalg.norm(v1)
    norm_v2 = v2 / np.linalg.norm(v2)

    cos_theta = np.dot(norm_v1.reshape((1, 3)), norm_v2)
    angle_rad = np.arccos(cos_theta)
    if angle_rad > np.pi / 2:
        angle_rad = np.pi - angle_rad

    if deg_result:
        # 将弧度转换为度
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    else:
        return angle_rad


# def calculate_refraction_vector(n_incident, n_normal, n_n1_n2):
#     """
#     计算折射方向向量
#     :param n_incident:
#     :param n_normal: 法向量，和入射光线同向
#     :param n_n1_n2: n_n1_n2 = n1/n2
#     :return:
#     """
#     n_incident = n_incident / np.linalg.norm(n_incident)
#     n_normal = n_normal / np.linalg.norm(n_normal)
#     cosi = np.dot(n_incident.reshape((1,3)), n_normal)
#     discriminant = 1.0 - n_n1_n2 ** 2 * (1 - cosi ** 2)
#     # n_refracted = np.zeros_like(n_incident)
#
#     if discriminant > 0:
#         # n_refracted[:] = n_relative * (n_incident + n_normal * dt) - n_normal * np.sqrt(discriminant)
#         n_refracted = n_n1_n2 * (n_incident - cosi[0, 0] * n_normal) - n_normal * np.sqrt(discriminant)
#         n_refracted = n_refracted / np.linalg.norm(n_refracted)
#         rad_incident = angle_between_vectors(n_incident, n_normal, deg_result=False)
#         rad_refracted = angle_between_vectors(n_refracted, n_normal, deg_result=False)
#         # print('入射角: {}, 出射角: {}, n2/n1: {}, sin(theta1)/sin(theta2): {}'.format(
#         #     np.rad2deg(rad_incident), np.rad2deg(rad_refracted), 1/n_n1_n2, np.sin(rad_incident)/np.sin(rad_refracted)
#         # ))
#         return n_refracted.reshape((3,1))
#     else:
#         raise ValueError('全反射，无法计算折射向量方向！\nn_n1_n2: {}, \nn_normal: {}, \nn_relative: {}, \n全反射角: {}, \n入射角: {}'.format(
#             n_incident.reshape(3,), n_normal.reshape(3,), n_n1_n2, np.rad2deg(np.arcsin(1/n_n1_n2)), angle_between_vectors(n_incident, n_normal)))
#         # return np.zeros((3,1))


def calculate_refraction_vector_old(n_incident, n_normal, n_n1_n2):
    """
    计算两个向量经过相对折射率为 n_relative 的平面后新的折射光线的方向
    :param n_incident: 入射光线方向
    :param n_normal: 折射面法向量
    :param n_n1_n2: 相对折射率
    :return:
    """
    n_refracted = np.array([1, 0, 0])
    # 在这里计算折射角度的逻辑
    cos_alpha_n1 = np.dot(n_incident.reshape((1, 3)), n_normal) / (
            np.linalg.norm(n_incident) * np.linalg.norm(n_normal))

    # 计算折射角度
    sin_alpha_wcs_n2 = math.sqrt(1 - np.squeeze(cos_alpha_n1) ** 2) * n_n1_n2  # 根据折射定律计算sin(α_wcs^l2)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # 此处可能会因为全反射的原因 sin 值大于 1, 无法计算失败，此处不做错误处理，但是需要注意
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    cos_alpha_wcs_n2 = math.sqrt(1 - sin_alpha_wcs_n2 ** 2)  # 根据三角关系计算cos(α_wcs^l2)

    # # 打印出对应的角度信息
    # alpha_wcs_n1 = math.acos(cos_alpha_n1)
    # alpha_wcs_l2 = math.acos(cos_alpha_wcs_n2)
    # print(np.rad2deg(alpha_wcs_n1))
    # print(np.rad2deg(alpha_wcs_l2))

    # 定义符号
    n_b, n_c = sp.symbols('n_b n_c')
    n_incident = n_incident.reshape((3,))
    n_normal = n_normal.reshape((3,))
    equation1 = sp.Eq(
        (n_incident[1] * n_normal[2] - n_incident[2] * n_normal[1]) * n_incident[0] + (
                n_incident[2] * n_normal[0] - n_incident[0] * n_normal[2]) * n_b + (
                n_incident[0] * n_normal[1] - n_incident[1] * n_normal[0]) * n_c, 0)
    equation2 = sp.Eq((n_refracted[0] * n_normal[0] + n_b * n_normal[1] + n_c * n_normal[2]) ** 2,
                      (n_refracted[0] ** 2 + n_b ** 2 + n_c ** 2) * (
                              n_normal[0] ** 2 + n_normal[1] ** 2 + n_normal[2] ** 2) * cos_alpha_wcs_n2 ** 2)
    # 解方程组
    solutions = sp.solve((equation1, equation2), (n_b, n_c))
    # 正常可以求出来两组解，接下来判断哪一组解是正确的
    n_refracted_1 = np.array([n_refracted[0], solutions[0][0], solutions[0][1]], dtype=float)
    n_refracted_2 = np.array([n_refracted[0], solutions[1][0], solutions[1][1]], dtype=float)
    norm_n1 = np.linalg.norm(n_incident)
    norm_n2_1 = np.linalg.norm(n_refracted_1)
    norm_n2_2 = np.linalg.norm(n_refracted_2)
    # 判断较小的夹角向量
    if np.abs(np.dot(n_incident, n_refracted_1) / (norm_n1 * norm_n2_1)) > np.abs(np.dot(n_incident, n_refracted_2) / (
            norm_n1 * norm_n2_2)):
        n_refracted = n_refracted_1
    else:
        n_refracted = n_refracted_2
    # print('n_refracted: ', n_refracted)
    return n_refracted.reshape(3, 1)


def calculate_line_plane_intersection_point(line_positions, line_vectors, A, B, C, D):
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
    # return_result = (line_positions.T + np.multiply(line_vectors.T, np.repeat(t.reshape((-1, 1)), 3, axis=1))).T

    return_result = (line_positions.T + np.multiply(line_vectors.T, np.tile(t[:, np.newaxis], (1, 3)))).T
    return return_result


# def calculate_line_surface_intersection_point(P1, V1, A, B, C, D):
#     return_result = np.zeros((3,1))
#     # 平面的系数
#     plane_vector = np.array([A, B, C]) # 计算平面的法线向量
#     vpt = np.dot(V1.reshape((1,3)), plane_vector)
#     # 平面上的一点
#     plane_point = np.zeros((3,1))
#     if A != 0:
#         plane_point[0] = -D / A
#     elif B != 0:
#         plane_point[1] = -D / B
#     elif C != 0:
#         plane_point[2] = -D / C
#     # 判断直线是否与平面平行
#     if vpt == 0:
#         return return_result
#     else:
#         t = np.dot((plane_point - P1).reshape((1,3)), plane_vector) / vpt
#         return_result[0] = P1[0] + V1[0] * t
#         return_result[1] = P1[1] + V1[1] * t
#         return_result[2] = P1[2] + V1[2] * t
#
#     return return_result

def calculate_line_surface_intersection_point_old(P1, V1, A, B, C, D):
    """
    计算直线和对应的平面的交点坐标
    :param P1:
    :param V1:
    :param A:
    :param B:
    :param C:
    :param D:
    :return:
    """
    V1 = V1.reshape((3,))
    P1 = P1.reshape((3,))
    x, y, z, t = sp.symbols('x y z t')
    equation1 = sp.Eq(x, V1[0] * t + P1[0])
    equation2 = sp.Eq(y, V1[1] * t + P1[1])
    equation3 = sp.Eq(z, V1[2] * t + P1[2])
    equation4 = sp.Eq(A * x + B * y + C * z + D, 0)
    # 解方程组
    solutions = sp.solve((equation1, equation2, equation3, equation4), (x, y, z, t))
    # print(solutions)
    # 返回对应的坐标信息
    return np.array([solutions[x], solutions[y], solutions[z]], dtype=float).reshape((3, 1))


def calculate_intersection_of_multi_lines(start_points, directions):
    '''
    计算距离多条空间直线距离最近的点的坐标
    strt_points: line start points; numpy array, nxdim
    directions: list dierctions; numpy array, nxdim
    return: the nearest points to n lines
    '''

    n, dim = start_points.shape

    G_left = np.tile(np.eye(dim), (n, 1))
    G_right = np.zeros((dim * n, n))

    for i in range(n):
        G_right[i * dim:(i + 1) * dim, i] = -directions[i, :]

    G = np.concatenate([G_left, G_right], axis=1)
    d = start_points.reshape((-1, 1))

    m = np.linalg.inv(np.dot(G.T, G)).dot(G.T).dot(d)

    # return m[0:dim]
    return m[:3].reshape(3).astype(np.float32)


def calculate_point_distance_to_multi_lines(point, line_positions, line_vectors):
    """
    计算一个点到很多空间直线的距离
    :param point: 3*1的 ndarray 点的坐标
    :param line_positions: 空间直线的起点
    :param line_vectors: 空间直线的方向向量
    :return:
    """
    vec1 = point - line_positions
    distance = np.linalg.norm(
        np.cross(vec1, line_vectors, axis=0), axis=0) / np.linalg.norm(line_vectors, axis=0)
    return distance


def calculate_line_line_intersection_or_closest_point(P1, V1, P2, V2, calc_distance=True):
    """
    计算两条直线的最近点和距离
    :param P1:
    :param V1:
    :param P2:
    :param V2:
    :return:
    """
    denominator = np.square(np.linalg.norm(np.cross(V1, V2, axis=0), axis=0))
    tmp_1 = (P2 - P1)
    tmp_2 = np.cross(V1, V2, axis=0)
    t1 = np.sum(np.multiply(np.cross(tmp_1, V2, axis=0), tmp_2), axis=0) / denominator
    t2 = np.sum(np.multiply(np.cross(tmp_1, V1, axis=0), tmp_2), axis=0) / denominator
    intersection_point_1 = P1 + t1 * V1
    intersection_point_2 = P2 + t2 * V2
    intersect_point = 0.5 * (intersection_point_1 + intersection_point_2)
    
    if calc_distance:
        return intersect_point, np.linalg.norm(intersection_point_2 - intersection_point_1, axis=0)
    else:
        return intersect_point, None


# def calculate_line_line_intersection_or_closest_point(P1, V1, P2, V2):
#     start_point = np.vstack([P1.reshape((-1, 3)), P2.reshape((-1, 3))])
#     directions = np.vstack([V1.reshape((-1, 3)), V2.reshape((-1, 3))])
#     intersection_point = intersection_of_multi_lines(start_point, directions).reshape((3,))
#
#     # 计算对应的空间距离之和
#     dis = 0
#     dis += point_distance_to_line(intersection_point, P1, V1)
#     dis += point_distance_to_line(intersection_point, P2, V2)
#
#     return intersection_point, dis


def coordinate_transformation(point_1, translation_matrix_1_to_2, rotate_matrix_1_to_2):
    """
    计算从一个坐标系到另一个坐标系下面的
    :param point_1:
    :param translation_matrix_1_to_2:
    :param rotate_matrix_1_to_2:
    :return:
    """
    point_2 = np.dot(rotate_matrix_1_to_2, point_1) + translation_matrix_1_to_2
    return point_2


class OptElement:
    def __init__(self,up_surface_params=None,down_surface_params=None,bound=[11.2,1],OptEl_to_world_translation_matrix=np.zeros((3,1)),\
                 OptEl_to_world_rotation_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]])):
        super().__init__()
        self.n1 = 1  # 入射空间的折射率
        self.n2 = 1.49  # 自身的折射率
        self.n3 = 1  # 出射空间的折射率
        self.up_surface_params = up_surface_params
        self.down_surface_params = down_surface_params
        self.bound_x = bound[0]
        self.bound_y = bound[1]
        
        self.OptEl_to_world_translation_matrix = OptEl_to_world_translation_matrix  
        self.OptEl_to_world_rotation_matrix = OptEl_to_world_rotation_matrix
        self.world_to_OptEl_translation_matrix = -self.OptEl_to_world_translation_matrix  # 光学器件的平移矩阵
        self.world_to_OptEl_rotation_matrix = self.OptEl_to_world_rotation_matrix.T  # 光学器件的旋转矩阵

        self.get_parameters()
    def get_parameters(self):
        """
        获取这个光学器件的位姿参数，包括旋转向量和平移向量
        注意：镜像的矩阵不会返回但是会更新
        """
        self.world_to_OptEl_rot_vec, self.world_to_OptEl_reflection = matrix_to_rotvec(self.world_to_OptEl_rotation_matrix)

        return self.world_to_OptEl_rot_vec, self.world_to_OptEl_translation_matrix

    def update_parameters(self, rot_vec, trans_vec, methord=TransformOrder.TRANSLATE_ROTATE):
        """
        更新位姿参数
        :param rot_vec:
        :param trans_vec:
        :param methord:
        """
        # 首先更新转换对应的旋转矩阵内容，然后调用原函数更新参数
        rot_mat = rotvec_to_matrix(rot_vec, self.world_to_OptEl_reflection)
        self.set_world_to_OptEl_rotate_translation_matrix(rot_mat, trans_vec, methord)

    def generate_lc_device_stl(self,output_filename, density=100):
        """
        根据LC_device实例生成一个闭合的实体STL模型。

        :param lc_device: LC_device类的一个实例。
        :param output_filename: 输出的STL文件名。
        :param density: XY平面的网格密度，数值越高模型越精细。
        """
        print(f"正在生成STL模型，密度为 {density}x{density}...")

        # --- 1. 生成顶点网格 ---
        x = np.linspace(0, self.bound_x, density)
        y = np.linspace(0, self.bound_y, density)
        xx, yy = np.meshgrid(x, y)

        # --- 2. 计算上下表面顶点 ---
        # 向量化计算所有Z坐标
        zz_up = self.up_surface_fun(xx, yy)
        zz_down = self.down_surface_fun(xx, yy)
        
        # 将顶点数据整合为 (density, density, 3) 的数组
        up_vertices = np.stack([xx, yy, zz_up], axis=-1)
        down_vertices = np.stack([xx, yy, zz_down], axis=-1)

        # --- 3. 构建上下表面三角面片 (向量化) ---
        def create_surface_faces(vertices, reverse_winding=False):
            """从顶点网格高效创建三角面片"""
            # (density-1, density-1) 个小方格
            quads_v1 = vertices[:-1, :-1]
            quads_v2 = vertices[1:, :-1]
            quads_v3 = vertices[:-1, 1:]
            quads_v4 = vertices[1:, 1:]

            num_quads = (density - 1) * (density - 1)
            faces1 = np.zeros((num_quads, 3, 3))
            faces2 = np.zeros((num_quads, 3, 3))

            # 第一个三角形 (v1, v2, v4)
            faces1[:, 0, :] = quads_v1.reshape(-1, 3)
            faces1[:, 1, :] = quads_v2.reshape(-1, 3)
            faces1[:, 2, :] = quads_v4.reshape(-1, 3)

            # 第二个三角形 (v1, v4, v3)
            faces2[:, 0, :] = quads_v1.reshape(-1, 3)
            faces2[:, 1, :] = quads_v4.reshape(-1, 3)
            faces2[:, 2, :] = quads_v3.reshape(-1, 3)
            
            if reverse_winding:
                # 翻转顶点顺序以使法向量朝外
                return np.concatenate([faces1[:, ::-1, :], faces2[:, ::-1, :]], axis=0)
            else:
                return np.concatenate([faces1, faces2], axis=0)

        top_faces = create_surface_faces(up_vertices)
        bottom_faces = create_surface_faces(down_vertices, reverse_winding=True)

        # --- 4. 构建侧壁三角面片 (向量化) ---
        def create_side_faces(top_v, bottom_v):
            """连接上下边界以创建侧壁"""
            all_side_faces = []
            # 遍历四条边: 右, 左, 上, 下
            boundaries = [
                (top_v[:, -1], bottom_v[:, -1]),   # 右边 (x=max)
                (top_v[::-1, 0], bottom_v[::-1, 0]), # 左边 (x=min), 反转顺序保持连接性
                (top_v[-1, ::-1], bottom_v[-1, ::-1]), # 上边 (y=max), 反转顺序
                (top_v[0, :], bottom_v[0, :])     # 下边 (y=min)
            ]

            for top_edge, bottom_edge in boundaries:
                edge_len = len(top_edge) - 1
                side_faces1 = np.zeros((edge_len, 3, 3))
                side_faces2 = np.zeros((edge_len, 3, 3))
                
                # 每个边上的小方格
                v1 = top_edge[:-1]
                v2 = bottom_edge[:-1]
                v3 = bottom_edge[1:]
                v4 = top_edge[1:]
                
                # 三角形1: (v1, v2, v3)
                side_faces1[:, 0, :] = v1
                side_faces1[:, 1, :] = v2
                side_faces1[:, 2, :] = v3

                # 三角形2: (v1, v3, v4)
                side_faces2[:, 0, :] = v1
                side_faces2[:, 1, :] = v3
                side_faces2[:, 2, :] = v4
                
                all_side_faces.append(side_faces1)
                all_side_faces.append(side_faces2)
            
            return np.concatenate(all_side_faces, axis=0)

        side_faces = create_side_faces(up_vertices, down_vertices)
        
        # --- 5. 整合所有面片并进行坐标变换 ---
        all_faces_local = np.concatenate([top_faces, bottom_faces, side_faces], axis=0)
        
        # 从 (num_faces, 3, 3) 变为 (num_faces * 3, 3) 以便进行矩阵运算
        num_faces_total = all_faces_local.shape[0]
        all_vertices_local = all_faces_local.reshape(num_faces_total * 3, 3)

        # 应用旋转和平移 (向量化)
        # world_coord = Rotation @ local_coord + Translation
        rotation_matrix = self.OptEl_to_world_rotation_matrix
        translation_vector = self.OptEl_to_world_translation_matrix
        
        # @ 是矩阵乘法运算符
        all_vertices_world = (rotation_matrix @ all_vertices_local.T).T + translation_vector.T

        # 将顶点数据重塑回 (num_faces, 3, 3)
        all_faces_world = all_vertices_world.reshape(num_faces_total, 3, 3)

        # --- 6. 创建并导出STL ---
        solid_mesh = mesh.Mesh(np.zeros(all_faces_world.shape[0], dtype=mesh.Mesh.dtype))
        solid_mesh.vectors = all_faces_world
        
        solid_mesh.save(output_filename)
        print(f"模型已成功保存至: {output_filename}")
        print(f"总面数: {num_faces_total}")
        def save_model(self):
            """
            保存模型的关键参数到 JSON 格式数据。
            :return: 返回 json 格式的数据
            """
            model_data = {}
            for attr, value in self.__dict__.items():
                # 如果属性是 numpy 数组，转换为列表
                if isinstance(value, np.ndarray):
                    model_data[attr] = value.tolist()
                else:
                    model_data[attr] = value
            return json.dumps(model_data, indent=4)

    def load_model(self, model_json):
        """
        从 JSON 数据加载模型的关键参数，设置到对象的属性中。
        :param model_json: JSON 格式的字符串，包含模型参数
        """
        model_data = json.loads(model_json)
        for attr, value in model_data.items():
            # 如果属性是列表并且期望为 numpy 数组，自动转换
            if isinstance(getattr(self, attr, None), np.ndarray):
                setattr(self, attr, np.array(value))
            else:
                setattr(self, attr, value)

    def set_world_to_OptEl_rotate_translation_matrix(self, rotate_matrix=None, translate_matrix=None,
                                                     method=TransformOrder.TRANSLATE_ROTATE):
        """
        设置从世界坐标系到本坐标系下的旋转平移矩阵，支持两种方法：
        1. TransformOrder.ROTATE_TRANSLATE: 先旋转后平移 (默认)
        2. TransformOrder.TRANSLATE_ROTATE: 先平移后旋转

        :param rotate_matrix: 从世界坐标系到本坐标系的旋转矩阵 (3x3)
        :param translate_matrix: 从世界坐标系到本坐标系的平移向量 (3x1 或 3,)
        :param method: 转换顺序，TransformOrder 枚举类型，默认为 ROTATE_TRANSLATE (先旋转后平移)
        :return: None
        """
        # 判断哪个需要更新
        if rotate_matrix is not None:
            # 检查旋转矩阵的形状是否为 3x3
            if rotate_matrix.shape != (3, 3):
                raise ValueError("rotate_matrix must be a 3x3 matrix")
            # 对旋转矩阵进行归一化处理
            rotate_matrix = rotate_matrix / np.linalg.norm(rotate_matrix, axis=1, keepdims=True)
            # 保存从世界坐标系到本坐标系的旋转和平移矩阵
            self.world_to_OptEl_rotation_matrix = rotate_matrix

        if translate_matrix is not None:
            # 检查平移矩阵的形状是否为 3x1 或 (3,)
            if translate_matrix.shape not in [(3, 1), (3,)]:
                raise ValueError("translate_matrix must be a 3x1 column vector or a 1D array of length 3")

            self.world_to_OptEl_translation_matrix = translate_matrix.reshape(3, 1)  # 确保平移向量为列向量

        # 计算逆变换（从本坐标系到世界坐标系）
        if method == TransformOrder.ROTATE_TRANSLATE:
            # 先旋转后平移
            self.OptEl_to_world_rotation_matrix = -self.world_to_OptEl_rotation_matrix.T
            self.OptEl_to_world_translation_matrix = -self.world_to_OptEl_translation_matrix
        elif method == TransformOrder.TRANSLATE_ROTATE:
            # 先平移后旋转
            self.OptEl_to_world_rotation_matrix = self.world_to_OptEl_rotation_matrix.T
            self.OptEl_to_world_translation_matrix = self.world_to_OptEl_rotation_matrix @ (
                -self.world_to_OptEl_translation_matrix)
        else:
            raise ValueError(
                f"Invalid method: {method}. Use TransformOrder.ROTATE_TRANSLATE or TransformOrder.TRANSLATE_ROTATE.")

    def OptEl_coordinate_to_world_coordinate(self, p_bcs, is_vector=False, order=TransformOrder.TRANSLATE_ROTATE):
        """
        将棱镜坐标中的点或者向量转换为世界坐标系
        :param p_bcs: 3*n ndarray, 要转换的点或者方向向量
        :param is_vector: 是否为方向向量，方向向量不受平移影响
        :param order: 转换顺序，TransformOrder 枚举类型，默认为 TRANSLATE_ROTATE (先平移后旋转)
        :return: 3*n ndarray, 转换后的点或向量
        """
        if is_vector:
            # 空间向量仅受旋转影响
            pl_wcs = np.dot(self.OptEl_to_world_rotation_matrix, p_bcs)
        else:
            if order == TransformOrder.TRANSLATE_ROTATE:
                # 先平移后旋转
                pl_wcs = np.dot(self.OptEl_to_world_rotation_matrix,
                                p_bcs + np.broadcast_to(self.OptEl_to_world_translation_matrix, p_bcs.shape))
            elif order == TransformOrder.ROTATE_TRANSLATE:
                # 先旋转后平移
                pl_wcs = (np.dot(self.OptEl_to_world_rotation_matrix, p_bcs) +
                          np.broadcast_to(self.OptEl_to_world_translation_matrix, p_bcs.shape))
            else:
                raise ValueError(
                    f"Invalid order: {order}. Use TransformOrder.TRANSLATE_ROTATE or TransformOrder.ROTATE_TRANSLATE.")
        return pl_wcs

    def world_coordinate_to_OptEl_coordinate(self, p_wcs, is_vector=False, order=TransformOrder.TRANSLATE_ROTATE):
        """
        将世界坐标中的点或者向量转换为棱镜坐标系
        :param p_wcs: 3*n ndarray, 要转换的点或者方向向量
        :param is_vector: 是否为方向向量，方向向量不受平移影响
        :param order: 转换顺序，TransformOrder 枚举类型，默认为 TRANSLATE_ROTATE (先平移后旋转)
        :return: 3*n ndarray, 转换后的点或向量
        """
        if is_vector:
            # 空间向量仅受旋转影响
            pl_bcs = np.dot(self.world_to_OptEl_rotation_matrix, p_wcs)
        else:
            if order == TransformOrder.TRANSLATE_ROTATE:
                # 先平移后旋转
                pl_bcs = np.dot(self.world_to_OptEl_rotation_matrix,
                                p_wcs + np.broadcast_to(self.world_to_OptEl_translation_matrix, p_wcs.shape))
            elif order == TransformOrder.ROTATE_TRANSLATE:
                # 先旋转后平移
                pl_bcs = (np.dot(self.world_to_OptEl_rotation_matrix, p_wcs) +
                          np.broadcast_to(self.world_to_OptEl_translation_matrix, p_wcs.shape))
            else:
                raise ValueError(
                    f"Invalid order: {order}. Use TransformOrder.TRANSLATE_ROTATE or TransformOrder.ROTATE_TRANSLATE.")
        return pl_bcs

    @abstractmethod
    def trace_ray(self, p_wcs, n_wcs, params=None):
        """
        追踪光线在 BFP 中的路径，输出内部坐标系下的光线位置和方向
        :param p_wcs:
        :param n_wcs:
        :param params: 其他参数
        :return:
        """
        pass

    def plot_OptEl(self, ax):
        """
        在指定的 ax 根据当前的位置绘制本光学器件，非必须
        :param ax:
        :return:
        """
        pass


class BP(OptElement):
    def __init__(self):
        """
        hello, world.
        """
        super().__init__()

        self.cot_alpha = 0
        self.prism_d = 0
        # 折射率
        self.n1 = 1  # 入射空间的折射率
        self.n2 = 1.5  # 自身的折射率
        self.n3 = 1  # 出射空间的折射率

    def trace_ray(self, p_wcs, n_wcs, params=None):
        """
        双棱镜光线追踪
        :param p_wcs:
        :param n_wcs:
        :param params: side=PrismSide.side_left
        :return:
        """
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        # n0_bcs = self.world_coordinate_to_bi_prism_coordinate(n_wcs, True)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, True)

        # ======================================================================
        # 自动计算相交平面判断光线相交的平面是哪个
        # if pl0_bcs[0, 0] > 0:
        if params == PrismSide.side_left:
            # -------------------------------------------------------------------
            # 左侧：平面 ADD'A'
            # 首先计算该直线和棱镜的哪个平面相交，通过x坐标的正负判断是否正确相交
            # pl1_bcs = calculate_line_plane_intersection_point(
            #     pl0_bcs, n1_bcs, 1, 0, self.cot_alpha.get(), 0)
            pl1_bcs = calculate_line_plane_intersection_point(
                pl0_bcs, n1_bcs, 1, 0, self.cot_alpha, 0)

            # 上面的运算结果是正确的，直接用上面计算的结果坐标
            # -------------------------------------------------------------------
            # 然后计算对应的光线的方向向量
            # n_adda = np.array([1, 0, self.cot_alpha])  # 平面 ADD'A' 的法向量
            n2_bcs = np.array([-1, 0, -self.cot_alpha]).reshape((3, 1))  # 平面 ADD'A' 的法向量
            # is_left = True
        else:
            # -------------------------------------------------------------------
            # 右侧：平面 BDD'B'
            # 上面的运算结果是错误的，需要重新计算平面交点
            # -------------------------------------------------------------------
            # 二次计算：计算 pl0_wcs 和第一个棱镜斜面的交点坐标(WCS)
            pl1_bcs = calculate_line_plane_intersection_point(
                pl0_bcs, n1_bcs, -1, 0, self.cot_alpha, 0)

            # 然后计算对应的光线的方向向量
            # n_dbbd = np.array([-1, 0, self.cot_alpha])  # 平面 DBB'D' 的法向量
            n2_bcs = np.array([1, 0, -self.cot_alpha]).reshape((3, 1))  # 平面 DBB'D' 的法向量
            # is_left = False

        # --------------------------------------------------------------------
        # 计算 pl1_bcs 点处的折射光线的方向向量
        n2_bcs = calculate_refraction_vector(
            n_1=self.n1, incident_direction=n1_bcs, n_2=self.n2,
            normal_direction=n2_bcs)  # 出射光线的方向向量，第一个数赋值为1，其他两个参数为计算结果

        # ------------------------------------------------------------------------
        # 计算和第二个面的交点坐标
        pl2_bcs = calculate_line_plane_intersection_point(pl1_bcs, n2_bcs, 0, 0, 1, -self.prism_d)
        # print(pl2_bcs)
        # -----------------------------------------------------------------
        # 计算 pl2_wcs 点处的折射光线的方向向量
        n_aabb = np.array([0, 0, -1]).reshape((3, 1))
        n3_bcs = calculate_refraction_vector(
            n_1=self.n2, incident_direction=n2_bcs, n_2=self.n3, normal_direction=n_aabb)

        # ------------------------------------------------------------------------
        # 将对应的棱镜坐标系的出射点及出射方向的向量转换为世界坐标系下的向量
        pl1_wcs = self.OptEl_coordinate_to_world_coordinate(pl1_bcs)
        pl2_wcs = self.OptEl_coordinate_to_world_coordinate(pl2_bcs)
        # pl3_wcs = self.prism_coordinate_to_world_coordinate(pl3_bcs)
        # pl4_wcs = self.prism_coordinate_to_world_coordinate(pl4_bcs)
        n2_wcs = self.OptEl_coordinate_to_world_coordinate(n2_bcs, True)
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_bcs, True)
        # n4_wcs = self.prism_coordinate_to_world_coordinate(n4_bcs, True)
        # n5_wcs = self.prism_coordinate_to_world_coordinate(n5_bcs, True)

        # if self.is_debugging:
        #     print('pl0_bcs: ', pl0_bcs)
        #     print('n1_bcs: ', n1_bcs)
        #     print('pl1_bcs: ', pl1_bcs)
        #     print('n2_bcs: ', n2_bcs)
        #     print('n3_wcs: ', n3_wcs)

        # return pl1_wcs, n3_wcs, pl2_wcs, n2_wcs, pl4_wcs, n5_wcs, pl3_wcs, n4_wcs
        return pl1_wcs, n3_wcs, pl2_wcs, n2_wcs


class BFP(OptElement):
    def __init__(self,up_surface_params=None,down_surface_params=None,bound=[11.2,1],OptEl_to_world_translation_matrix=np.zeros((3,1)),\
                 OptEl_to_world_rotation_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]])):
        # 将构造函数收到的参数全部传递给父类
        super().__init__(up_surface_params=up_surface_params,
                        down_surface_params=down_surface_params,
                        bound=bound,
                        OptEl_to_world_translation_matrix=OptEl_to_world_translation_matrix,
                        OptEl_to_world_rotation_matrix=OptEl_to_world_rotation_matrix)
        self.left_surface_params = np.array([
            17.595166825667565,
            -0.752562775772513,
            0.0022358592685913824,
            0.005986523093043514,
            0.002862307511814188,
            0.0025178751147601135,
            -0.00026641608722236654,
            -0.0004363048747108025,
            0.00011895613233355662,
            7.869202300579768e-06,
            2.4405962614246697e-05,
            -1.855449678101475e-05
        ])
        # 上表面位置（作为基面为0）
        self.up_surface_z = 0
        # 不同介质的折射率
        self.n1 = 1
        self.n2 = 1.49
        self.n3 = 1

    def plot_OptEl(self, ax):
        # 定义 -11 到 11 的 x, y 区域
        x_count, y_count = 20, 20
        x = np.linspace(-11, 11, x_count)  # x 方向 100 个点
        y = np.linspace(-11, 11, y_count)  # y 方向 100 个点
        x, y = np.meshgrid(x, y)  # 创建网格

        # 计算对应的 z 值
        z1 = np.zeros_like(x)
        z2 = self.full_surface_fun(x, y)
        # 转换为 10000x3 的 ndarray
        up_surface_points = np.vstack((x.ravel(), y.ravel(), z1.ravel()))
        side_surface_points = np.vstack((x.ravel(), y.ravel(), z2.ravel()))
        # 转换后的坐标位置
        side_surface_points_transformed = self.world_coordinate_to_OptEl_coordinate(side_surface_points, False)
        up_surface_points_transformed = self.world_coordinate_to_OptEl_coordinate(up_surface_points, False)

        # 先绘制自由曲面
        ax.plot_surface(
            side_surface_points_transformed[0, :].reshape(x_count, y_count),
            side_surface_points_transformed[1, :].reshape(x_count, y_count),
            side_surface_points_transformed[2, :].reshape(x_count, y_count),
            cmap='viridis', edgecolor='none')
        # 然后绘制上表面
        ax.plot_surface(
            up_surface_points_transformed[0, :].reshape(x_count, y_count),
            up_surface_points_transformed[1, :].reshape(x_count, y_count),
            up_surface_points_transformed[2, :].reshape(x_count, y_count),
            cmap='viridis', edgecolor='none')

    def trace_ray(self, p_wcs, n_wcs, params=None):
        """
        追踪光线在 BFP 中的路径，输出内部坐标系下的光线位置和方向
        :param p_wcs:
        :param n_wcs:
        :param params:
        :return:
        """
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        # n0_bcs = self.world_coordinate_to_bi_prism_coordinate(n_wcs, True)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, True)

        # ======================================================================
        # 自动计算相交平面判断光线相交的平面是哪个
        # 此处经过验证刚好应该是反的
        if params == PrismSide.side_right:
            # -------------------------------------------------------------------
            # 左侧：
            # pl1_bcs = calculate_line_plane_intersection_point(
            #     pl0_bcs, n1_bcs, 1, 0, self.cot_alpha, 0)
            _, pl1_bcs = BFP_calculate_line_surface_intersection_points_by_binary_search(
                pl0_bcs, n1_bcs, self.left_surface_fun)
            n_BFP_side = self.left_surface_gradient_fun(pl1_bcs[0, :], pl1_bcs[1, :], )

            # 上面的运算结果是正确的，直接用上面计算的结果坐标
            # -------------------------------------------------------------------
            # 然后计算对应的光线的方向向量
            # n_adda = np.array([1, 0, self.cot_alpha])  # 平面 ADD'A' 的法向量
            # n2_bcs = np.array([-1, 0, -self.cot_alpha]).reshape((3, 1))  # 平面 ADD'A' 的法向量

            # is_left = True
        else:
            # -------------------------------------------------------------------
            # 右侧：平面 BDD'B'
            # 上面的运算结果是错误的，需要重新计算平面交点
            # -------------------------------------------------------------------
            # 二次计算：计算 pl0_wcs 和第一个棱镜斜面的交点坐标(WCS)
            # pl1_bcs = calculate_line_plane_intersection_point(
            #     pl0_bcs, n1_bcs, -1, 0, self.cot_alpha, 0)
            _, pl1_bcs = BFP_calculate_line_surface_intersection_points_by_binary_search(
                pl0_bcs, n1_bcs, self.right_surface_fun)
            n_BFP_side = self.right_surface_gradient_fun(pl1_bcs[0, :], pl1_bcs[1, :])

            # 然后计算对应的光线的方向向量
            # n_dbbd = np.array([-1, 0, self.cot_alpha])  # 平面 DBB'D' 的法向量
            # n2_bcs = np.array([1, 0, -self.cot_alpha]).reshape((3, 1))  # 平面 DBB'D' 的法向量
            # is_left = False

        # --------------------------------------------------------------------
        # 计算 pl1_bcs 点处的折射光线的方向向量
        n2_bcs = calculate_refraction_vector(
            n_1=self.n1, incident_direction=n1_bcs, n_2=self.n2,
            normal_direction=n_BFP_side)  # 出射光线的方向向量，第一个数赋值为1，其他两个参数为计算结果

        # result = check_snell_law(self.n1, n1_bcs, self.n2, n_BFP_side, n2_bcs)

        # ------------------------------------------------------------------------
        # 计算和第二个面的交点坐标
        pl2_bcs = calculate_line_plane_intersection_point(
            pl1_bcs, n2_bcs, 0, 0, 1, self.up_surface_z)
        # print(pl2_bcs)
        # -----------------------------------------------------------------
        # 计算 pl2_wcs 点处的折射光线的方向向量
        n_aabb = np.array([0, 0, 1]).reshape((3, 1))
        n3_bcs = calculate_refraction_vector(
            n_1=self.n2, incident_direction=n2_bcs, n_2=self.n3, normal_direction=n_aabb)

        # result = check_snell_law(self.n2, n2_bcs, self.n3, n_aabb, n3_bcs)

        # ------------------------------------------------------------------------
        # 将对应的棱镜坐标系的出射点及出射方向的向量转换为世界坐标系下的向量
        pl1_wcs = self.OptEl_coordinate_to_world_coordinate(pl1_bcs)
        pl2_wcs = self.OptEl_coordinate_to_world_coordinate(pl2_bcs)
        # pl3_wcs = self.prism_coordinate_to_world_coordinate(pl3_bcs)
        # pl4_wcs = self.prism_coordinate_to_world_coordinate(pl4_bcs)
        n2_wcs = self.OptEl_coordinate_to_world_coordinate(n2_bcs, True)
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_bcs, True)
        # n4_wcs = self.prism_coordinate_to_world_coordinate(n4_bcs, True)
        # n5_wcs = self.prism_coordinate_to_world_coordinate(n5_bcs, True)

        # if self.is_debugging:
        #     print('pl0_bcs: ', pl0_bcs)
        #     print('n1_bcs: ', n1_bcs)
        #     print('pl1_bcs: ', pl1_bcs)
        #     print('n2_bcs: ', n2_bcs)
        #     print('n3_wcs: ', n3_wcs)

        # return pl1_wcs, n3_wcs, pl2_wcs, n2_wcs, pl4_wcs, n5_wcs, pl3_wcs, n4_wcs
        return pl2_wcs, n3_wcs, pl1_wcs, n2_wcs

    def left_surface_fun(self, x, y):
        """
        左侧 BFP 的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (n,) 的 ndarray, 表示该点的 z 坐标
        """
        z = (
                self.left_surface_params[0] +
                self.left_surface_params[1] * x +
                self.left_surface_params[2] * x ** 2 +
                self.left_surface_params[3] * y ** 2 +
                self.left_surface_params[4] * x ** 3 +
                self.left_surface_params[5] * x * y ** 2 +
                self.left_surface_params[6] * x ** 4 +
                self.left_surface_params[7] * x ** 2 * y ** 2 +
                self.left_surface_params[8] * y ** 4 +
                self.left_surface_params[9] * x ** 5 +
                self.left_surface_params[10] * x ** 3 * y ** 2 +
                self.left_surface_params[11] * x * y ** 4
        )
        return z

    def right_surface_fun(self, x, y):
        """
        右侧 BFP 的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (n,) 的 ndarray, 表示该点的 z 坐标
        """
        return self.left_surface_fun(-x, y)

    def full_surface_fun(self, x, y):
        """
        计算整个对称的面型 (左侧与右侧合并)
        根据 x 的正负自动选择左表面或右表面
        """
        z = np.where(x > 0, self.left_surface_fun(x, y), self.right_surface_fun(x, y))
        return z

    def left_surface_gradient_fun(self, x: np.ndarray, y: np.ndarray, dir=1):
        """
        左侧 BFP 微分的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (3, n) 的 ndarray, 表示该点的梯度方向
        """
        dz_dx = (
                self.left_surface_params[1] +
                2 * self.left_surface_params[2] * x +
                3 * self.left_surface_params[4] * x ** 2 +
                self.left_surface_params[5] * y ** 2 +
                4 * self.left_surface_params[6] * x ** 3 +
                2 * self.left_surface_params[7] * x * y ** 2 +
                5 * self.left_surface_params[9] * x ** 4 +
                3 * self.left_surface_params[10] * x ** 2 * y ** 2 +
                self.left_surface_params[11] * y ** 4
        )

        dz_dy = (
                2 * self.left_surface_params[3] * y +
                2 * self.left_surface_params[5] * x * y +
                2 * self.left_surface_params[7] * x ** 2 * y +
                4 * self.left_surface_params[8] * y ** 3 +
                2 * self.left_surface_params[10] * x ** 3 * y +
                4 * self.left_surface_params[11] * x * y ** 3
        )

        gradient = np.vstack((-dz_dx, -dz_dy, np.ones_like(x) * dir))
        return gradient

    def right_surface_gradient_fun(self, x, y, dir=1):
        """
        右侧 BFP 微分的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (3, n) 的 ndarray, 表示该点的梯度方向
        """
        neg_x_gradient = self.left_surface_gradient_fun(-x, y, dir=dir)
        neg_x_gradient[0] *= -1  # 反转 x 梯度的符号
        return neg_x_gradient

    def full_surface_gradient_fun(self, x, y):
        """
        计算整个对称的面型 (左侧与右侧合并)
        根据 x 的正负自动选择左表面或右表面
        """
        z = np.where(x > 0, self.left_surface_gradient_fun(x, y), self.right_surface_gradient_fun(x, y))
        return z


def check_snell_law(n_1, incident_direction, n_2, normal_direction, refraction_direction):
    # 计算入射光线和法线的夹角 sin(theta_1)
    incident_direction = incident_direction / np.linalg.norm(incident_direction)
    normal_direction = normal_direction / np.linalg.norm(normal_direction)
    refraction_direction = refraction_direction / np.linalg.norm(refraction_direction)
    # 计算正弦值
    sin_in = np.linalg.norm(np.cross(incident_direction.T, normal_direction.T))
    sin_refrac = np.linalg.norm(np.cross(refraction_direction.T, normal_direction.T))

    return np.isclose(n_2 / n_1, sin_in / sin_refrac, atol=1e-6)


class PMMA(OptElement):
    def __init__(self):
        super().__init__()
        self.n1 = 1  # 空气折射率
        self.n2 = 1.49  # PMMA 折射率
        self.n3 = 1.5  # PDMS 折射率

        self.thickness = 2

    def plot_OptEl(self, ax):
        x_count, y_count = 2, 2
        x = np.linspace(-11, 11, x_count)
        y = np.linspace(-11, 11, y_count)
        x, y = np.meshgrid(x, y)
        z1 = np.ones_like(x) * self.thickness
        z2 = np.zeros_like(x)
        # 转换为 3*n 的 ndarray
        surface1_points = np.vstack((x.ravel(), y.ravel(), z1.ravel()))
        surface2_points = np.vstack((x.ravel(), y.ravel(), z2.ravel()))
        surface1_points_transformed = self.OptEl_coordinate_to_world_coordinate(surface1_points, False)
        surface2_points_transformed = self.OptEl_coordinate_to_world_coordinate(surface2_points, False)

        ax.plot_surface(
            surface1_points_transformed[0, :].reshape(x_count, y_count),
            surface1_points_transformed[1, :].reshape(x_count, y_count),
            surface1_points_transformed[2, :].reshape(x_count, y_count),
            cmap='viridis', edgecolor='none')
        ax.plot_surface(
            surface2_points_transformed[0, :].reshape(x_count, y_count),
            surface2_points_transformed[1, :].reshape(x_count, y_count),
            surface2_points_transformed[2, :].reshape(x_count, y_count),
            cmap='viridis', edgecolor='none')

    def trace_ray(self, p_wcs, n_wcs, params=None):
        """
        追踪光线到PDMS内部
        :param p_wcs:
        :param n_wcs:
        :param params:
        :return:
        """
        pl2_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        n3_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, True)

        # ----------------------------------------------------------
        # 3. 光线经过一段空气到达 PMMA 板
        pl3_bcs = calculate_line_plane_intersection_point(
            pl2_bcs, n3_bcs, 0, 0, 1, 0)

        # 4. 从空气进入到PMMA板
        # 计算经过 PMMA 的折射后的光线方向向量
        n_acrylic = np.array([0, 0, -1]).reshape((3, 1))
        # n4_wcs = calculate_refraction_vector(n3_wcs, n_acrylic, 1/self.params.n)
        n4_bcs = calculate_refraction_vector(
            n_1=self.n1, incident_direction=n3_bcs, n_2=self.n2, normal_direction=n_acrylic)

        # 5. 从 PMMA 板到 PDMS
        pl4_bcs = calculate_line_plane_intersection_point(
            pl3_bcs, n4_bcs, 0, 0, 1, -self.thickness)
        n5_bcs = calculate_refraction_vector(
            n_1=self.n2, incident_direction=n4_bcs, n_2=self.n3, normal_direction=n_acrylic)

        # ------------------------------------------------------------------------
        # 将对应的棱镜坐标系的出射点及出射方向的向量转换为世界坐标系下的向量
        pl3_wcs = self.OptEl_coordinate_to_world_coordinate(pl3_bcs)
        pl4_wcs = self.OptEl_coordinate_to_world_coordinate(pl4_bcs)
        n4_wcs = self.OptEl_coordinate_to_world_coordinate(n4_bcs, True)
        n5_wcs = self.OptEl_coordinate_to_world_coordinate(n5_bcs, True)

        return pl4_wcs, n5_wcs, pl3_wcs, n4_wcs

class GLASS(PMMA):
    def __init__(self):
        super.__init__()
        self.n3=1

class LC_device(BFP):
    def __init__(self,up_surface_params,down_surface_params,bound=[11.2,1],OptEl_to_world_translation_matrix=np.zeros((3,1)),\
                 OptEl_to_world_rotation_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]])):
        # 将构造函数收到的参数全部传递给父类
        super().__init__(up_surface_params=up_surface_params,
                        down_surface_params=down_surface_params,
                        bound=bound,
                        OptEl_to_world_translation_matrix=OptEl_to_world_translation_matrix,
                        OptEl_to_world_rotation_matrix=OptEl_to_world_rotation_matrix)
        self.n1 = 1  # 入射空间的折射率
        self.n2 = 1.49  # 自身的折射率
        self.n3 = 1  # 出射空间的折射率
        self.up_surface_params = up_surface_params
        self.down_surface_params = down_surface_params
        self.bound_x = bound[0]
        self.bound_y = bound[1]
        
        self.OptEl_to_world_translation_matrix = OptEl_to_world_translation_matrix  
        self.OptEl_to_world_rotation_matrix = OptEl_to_world_rotation_matrix
        self.world_to_OptEl_translation_matrix = -self.OptEl_to_world_translation_matrix  # 光学器件的平移矩阵
        self.world_to_OptEl_rotation_matrix = self.OptEl_to_world_rotation_matrix.T  # 光学器件的旋转矩阵
    def up_surface_fun(self, x, y):
        """
        上侧 LC 的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (n,) 的 ndarray, 表示该点的 z 坐标
        """
        z = (
                self.up_surface_params[0] +
                self.up_surface_params[1] * x +
                self.up_surface_params[2] * x ** 2 +
                self.up_surface_params[3] * y ** 2 +
                self.up_surface_params[4] * x ** 3 +
                self.up_surface_params[5] * x * y ** 2 +
                self.up_surface_params[6] * x ** 4 +
                self.up_surface_params[7] * x ** 2 * y ** 2 +
                self.up_surface_params[8] * y ** 4 +
                self.up_surface_params[9] * x ** 5 +
                self.up_surface_params[10] * x ** 3 * y ** 2 +
                self.up_surface_params[11] * x * y ** 4
        )
        return z
    def down_surface_fun(self, x, y):
        """
        下侧 LC 的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (n,) 的 ndarray, 表示该点的 z 坐标
        """
        z = (
                self.down_surface_params[0] +
                self.down_surface_params[1] * x +
                self.down_surface_params[2] * x ** 2 +
                self.down_surface_params[3] * y ** 2 +
                self.down_surface_params[4] * x ** 3 +
                self.down_surface_params[5] * x * y ** 2 +
                self.down_surface_params[6] * x ** 4 +
                self.down_surface_params[7] * x ** 2 * y ** 2 +
                self.down_surface_params[8] * y ** 4 +
                self.down_surface_params[9] * x ** 5 +
                self.down_surface_params[10] * x ** 3 * y ** 2 +
                self.down_surface_params[11] * x * y ** 4
        )
        return z
    def up_surface_gradient_fun(self, x: np.ndarray, y: np.ndarray, dir=1):
        """
        上侧 LC 微分的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (3, n) 的 ndarray, 表示该点的梯度方向
        """
        dz_dx = (
                self.up_surface_params[1] +
                2 * self.up_surface_params[2] * x +
                3 * self.up_surface_params[4] * x ** 2 +
                self.up_surface_params[5] * y ** 2 +
                4 * self.up_surface_params[6] * x ** 3 +
                2 * self.up_surface_params[7] * x * y ** 2 +
                5 * self.up_surface_params[9] * x ** 4 +
                3 * self.up_surface_params[10] * x ** 2 * y ** 2 +
                self.up_surface_params[11] * y ** 4
        )

        dz_dy = (
                2 * self.up_surface_params[3] * y +
                2 * self.up_surface_params[5] * x * y +
                2 * self.up_surface_params[7] * x ** 2 * y +
                4 * self.up_surface_params[8] * y ** 3 +
                2 * self.up_surface_params[10] * x ** 3 * y +
                4 * self.up_surface_params[11] * x * y ** 3
        )

        gradient = np.vstack((-dz_dx, -dz_dy, np.ones_like(x) * dir))
        return gradient
    def down_surface_gradient_fun(self, x, y, dir=1):
        """
        下侧 LC 微分的函数表达式
        :param x: (n,) 的 ndarray
        :param y: (n,) 的 ndarray
        :return: (3, n) 的 ndarray, 表示该点的梯度方向
        """
        dz_dx = (
                self.down_surface_params[1] +
                2 * self.down_surface_params[2] * x +
                3 * self.down_surface_params[4] * x ** 2 +
                self.down_surface_params[5] * y ** 2 +
                4 * self.down_surface_params[6] * x ** 3 +
                2 * self.down_surface_params[7] * x * y ** 2 +
                5 * self.down_surface_params[9] * x ** 4 +
                3 * self.down_surface_params[10] * x ** 2 * y ** 2 +
                self.down_surface_params[11] * y ** 4
        )

        dz_dy = (
                2 * self.down_surface_params[3] * y +
                2 * self.down_surface_params[5] * x * y +
                2 * self.down_surface_params[7] * x ** 2 * y +
                4 * self.down_surface_params[8] * y ** 3 +
                2 * self.down_surface_params[10] * x ** 3 * y +
                4 * self.down_surface_params[11] * x * y ** 3
        )

        gradient = np.vstack((-dz_dx, -dz_dy, np.ones_like(x) * dir))
        return gradient
    def plot_OptEl(self, ax, resolution=20):
        """
        在给定的3D坐标轴上绘制光学元件的表面。

        :param ax: Matplotlib 的 3D 坐标轴对象。
        :param x_range: tuple, X方向的绘图范围 (min, max)。
        :param y_range: tuple, Y方向的绘图范围 (min, max)。
        :param resolution: int, 每个方向上的网格点数（精度）。
        """
        # 使用传入的参数来定义 x, y 区域
        x = np.linspace(0,self.bound_x, resolution)
        y = np.linspace(0,self.bound_y, resolution)
        x, y = np.meshgrid(x, y) # 创建网格

        # 计算对应的 z 值
        up_surface_z = self.up_surface_fun(x, y)
        down_surface_z = self.down_surface_fun(x, y)
        # 转换为 10000x3 的 ndarray
        up_surface_points = np.vstack((x.ravel(), y.ravel(), up_surface_z.ravel()))
        down_surface_points = np.vstack((x.ravel(), y.ravel(), down_surface_z.ravel()))
        # 转换后的坐标位置
        up_surface_points_transformed = self.OptEl_coordinate_to_world_coordinate(up_surface_points, False)
        down_surface_points_transformed = self.OptEl_coordinate_to_world_coordinate(down_surface_points, False)
        # 先绘制自由曲面
        ax.plot_surface(
            up_surface_points_transformed[0, :].reshape(resolution, resolution),
            up_surface_points_transformed[1, :].reshape(resolution, resolution),
            up_surface_points_transformed[2, :].reshape(resolution, resolution),
            cmap='viridis', edgecolor='none')
        ax.plot_surface(
            down_surface_points_transformed[0, :].reshape(resolution, resolution),
            down_surface_points_transformed[1, :].reshape(resolution, resolution),
            down_surface_points_transformed[2, :].reshape(resolution, resolution),
            cmap='viridis', edgecolor='none')
    def trace_ray(self, p_wcs, n_wcs):
        """
        追踪光线在 LC 中的路径。
        修改版逻辑：保留所有通过第一个面的光线。
        如果光线未击中第二个面，则将其方向向量延长，并报告其最终状态。
        :param p_wcs:
        :param n_wcs:
        :return: pl2_wcs, n3_wcs, pl1_wcs, n2_wcs, final_mask
        """
        num_rays_initial = p_wcs.shape[1]

        # 坐标变换
        pl0_bcs = self.world_coordinate_to_OptEl_coordinate(p_wcs)
        n1_bcs = self.world_coordinate_to_OptEl_coordinate(n_wcs, True)

        # ====================================================================
        # 第一步: 计算与第一个面的交点并进行第一次筛选
        # ====================================================================
        _, pl1_bcs_all = calculate_line_surface_intersection_points_by_binary_search(
            pl0_bcs, n1_bcs, self.down_surface_fun, min_range=-100, max_range=100
        )
        
        # 筛选通过第一个边界的光线 (这些是所有需要保留的光线)
        pl1_after_filter1, (n1_after_filter1,), mask1 = filter_rays_by_boundary(
            pl1_bcs_all, (0, self.bound_x), (0, self.bound_y), n1_bcs
        )

        # 如果没有任何光线击中第一个面，直接返回
        if pl1_after_filter1.shape[1] == 0:
            print("Warning: No rays hit the valid area of the first surface.")
            empty_arr = np.array([]).reshape(3, 0)
            final_mask = np.full(num_rays_initial, False, dtype=bool)
            return empty_arr, empty_arr, empty_arr, empty_arr, final_mask

        # 计算第一次折射
        n_LC_down_side = self.down_surface_gradient_fun(pl1_after_filter1[0, :], pl1_after_filter1[1, :])
        n2_after_filter1 = calculate_refraction_vector(
            n_1=self.n1, incident_direction=n1_after_filter1, n_2=self.n2,
            normal_direction=n_LC_down_side
        )

        # ====================================================================
        # ## 核心修改部分开始 ##
        # ====================================================================
        # 第二步: 计算与第二个面的交点，并“分类”光线，而不是“筛选”
        # ====================================================================
        _, pl2_bcs_from_valid_1 = calculate_line_surface_intersection_points_by_binary_search(
            pl1_after_filter1, n2_after_filter1, self.up_surface_fun, min_range=-100, max_range=100
        )
        
        # 获取一个布尔掩码，用于区分击中和错过的光线
        # 注意：这里我们只需要掩码，所以忽略返回的其他数组
        _, _, mask_hit_surface2 = filter_rays_by_boundary(
            pl2_bcs_from_valid_1, (0, self.bound_x), (0, self.bound_y)
        )
        mask_miss_surface2 = ~mask_hit_surface2
        
        # 创建空的数组用于存放最终合并的结果
        num_valid_rays_1 = pl1_after_filter1.shape[1]
        pl2_bcs_combined = np.zeros((3, num_valid_rays_1))
        n3_bcs_combined = np.zeros((3, num_valid_rays_1))

        # --------------------------------------------------------------------
        # 分组处理 A: 成功击中第二个面的光线 (原始逻辑)
        # --------------------------------------------------------------------
        if np.any(mask_hit_surface2):
            # 提取出击中的光线
            pl2_hit = pl2_bcs_from_valid_1[:, mask_hit_surface2]
            n2_hit = n2_after_filter1[:, mask_hit_surface2]
            
            # 计算最终出射光线的法向量和方向
            n_LC_up_side = self.up_surface_gradient_fun(pl2_hit[0, :], pl2_hit[1, :])
            n3_hit = calculate_reflection_vector(
                incident_direction=n2_hit,
                normal_direction=n_LC_up_side
            )
            
            # 将处理结果存入合并数组的对应位置
            pl2_bcs_combined[:, mask_hit_surface2] = pl2_hit
            n3_bcs_combined[:, mask_hit_surface2] = n3_hit

        # --------------------------------------------------------------------
        # 分组处理 B: 未击中第二个面的光线 (新逻辑)
        # --------------------------------------------------------------------
        if np.any(mask_miss_surface2):
            # 提取出错过的光线
            pl1_miss = pl1_after_filter1[:, mask_miss_surface2]
            n2_miss = n2_after_filter1[:, mask_miss_surface2]
            
            # 按照新规则计算最终状态
            # 最终位置 = 第一个面交点 + 延伸方向 * 距离
            pl2_miss = pl1_miss + n2_miss * 30
            # 最终方向 = 第一次折射后的方向
            n3_miss = n2_miss
            
            # 将处理结果存入合并数组的对应位置
            pl2_bcs_combined[:, mask_miss_surface2] = pl2_miss
            n3_bcs_combined[:, mask_miss_surface2] = n3_miss
            
        # ====================================================================
        # ## 核心修改部分结束 ##
        # ====================================================================

        # 第三步: 生成最终掩码 (现在它代表所有通过第一个面的光线)
        final_mask = mask1
        
        # ====================================================================
        # 第四步: 坐标变换并返回紧凑的结果
        # ====================================================================
        # 此时，pl1_after_filter1, n2_after_filter1, pl2_bcs_combined, n3_bcs_combined 的光线数和顺序完全一致
        pl1_wcs = self.OptEl_coordinate_to_world_coordinate(pl1_after_filter1)
        n2_wcs = self.OptEl_coordinate_to_world_coordinate(n2_after_filter1, True)
        pl2_wcs = self.OptEl_coordinate_to_world_coordinate(pl2_bcs_combined)
        n3_wcs = self.OptEl_coordinate_to_world_coordinate(n3_bcs_combined, True)
        
        return pl2_wcs, n3_wcs, pl1_wcs, n2_wcs, final_mask
    

class Point_light_source(OptElement):
    """
    定义一个三维空间中的点光源。
    
    该光源从一个指定位置，向一个半球形或锥形区域内均匀发射光线。
    """
    def __init__(self, 
                num_rays=1000000, 
                emission_angle_deg=90, 
                color='red',
                OptEl_to_world_translation_matrix=np.zeros((3,1)),
                OptEl_to_world_rotation_matrix=np.array([[1,0,0],[0,1,0],[0,0,1]])):
        # 将构造函数收到的参数全部传递给父类
        super().__init__(OptEl_to_world_translation_matrix=OptEl_to_world_translation_matrix,
                        OptEl_to_world_rotation_matrix=OptEl_to_world_rotation_matrix)
        self.num_rays = num_rays  # 光线数量
        self.emission_angle_deg = emission_angle_deg  # 发射角度（以度为单位）
        self.color=color  # 光源颜色
        self.OptEl_to_world_translation_matrix = OptEl_to_world_translation_matrix  
        self.OptEl_to_world_rotation_matrix = OptEl_to_world_rotation_matrix
        self.world_to_OptEl_translation_matrix = -self.OptEl_to_world_translation_matrix  # 光学器件的平移矩阵
        self.world_to_OptEl_rotation_matrix = self.OptEl_to_world_rotation_matrix.T  # 光学器件的旋转矩阵

    def plot_OptEl(self, ax):
        """
        在给定的 3D Matplotlib 坐标轴上绘制点光源。
        """
        # 将元件坐标转换为世界坐标进行绘制
        point = self.OptEl_to_world_translation_matrix
        ax.scatter(point[0], point[1], point[2], color=self.color, s=50, label='Point Light Source')

    def trace_ray(self):
        """
        生成从点光源发出的所有光线的位置和方向向量。
        
        为了在球体或球冠上实现均匀分布，我们使用特定的数学方法，
        而不是简单地随机选择球坐标的两个角度。

        :return: (p_wcs, n_wcs)
                 p_wcs: np.ndarray, 所有光线的起始位置，形状为 (3, num_rays)。
                 n_wcs: np.ndarray, 所有光线的方向向量（已归一化），形状为 (3, num_rays)。
        """
        # 1. 生成光线位置
        # 所有光线都从同一点发出，因此我们将位置向量复制 num_rays 次。
        p_wcs = np.tile(self.OptEl_to_world_translation_matrix, (1, self.num_rays))

        # 2. 生成光线方向向量
        # 在由 emission_angle_deg 定义的球冠上均匀生成随机方向
        
        # phi 是方位角，在 [0, 2*pi] 之间均匀分布
        phi = np.random.uniform(0, 2 * np.pi, self.num_rays)
        
        # cos_theta 在 [cos(emission_angle), 1] 之间均匀分布，以确保球面均匀性
        cos_theta_max = np.cos(np.deg2rad(self.emission_angle_deg))
        cos_theta = np.random.uniform(cos_theta_max, 1, self.num_rays)
        
        # 计算 theta 和 sin_theta
        theta = np.arccos(cos_theta)
        sin_theta = np.sin(theta)
        
        # 从球坐标转换为笛卡尔坐标 (nx, ny, nz)
        nx = sin_theta * np.cos(phi)
        ny = sin_theta * np.sin(phi)
        nz = cos_theta # nz 直接就是 cos_theta

        # 将方向向量组合成一个 (3, num_rays) 的数组
        n_wcs = np.vstack((nx, ny, nz))
        
        # 尽管从数学上讲向量已经是归一化的，但进行一次确认总是一个好习惯
        # norms = np.linalg.norm(n_wcs, axis=0)
        # n_wcs = n_wcs / norms

        n_wcs = self.OptEl_coordinate_to_world_coordinate(n_wcs, True)

        return p_wcs, n_wcs
