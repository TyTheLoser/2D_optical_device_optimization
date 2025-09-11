import copy
import json
import os.path
import pickle
import time
import warnings
from abc import abstractmethod
from enum import Enum

import cv2
# import utils

import numpy as np
# import cupy as np

import math
from typing import Tuple
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

from utils.utils_camera_calibration import BiPrismCameraCalibration, CameraParams
from utils.bi_prism_data_analyzer import BiPrismDataVisualization
from utils.utils_error_analysis import plot_real_points_and_calc_points
from utils.utils_free_form_prism_design import binary_search, calculate_refraction_vector
from utils.utils_ray_trace import *


# from utils_free_form_prism_design import *


class BiPrismParamsTrans:
    """
    棱镜成像系统固定可调参数结构体
    """

    def __init__(self):
        # 系统结构参数
        self.ccd_size_pixel = np.array([1920, 1080])  # ccd pixel size
        self.intrinsic_matrix = np.zeros((3, 3))
        # self.L_ol = 50

        # 棱镜位置坐标变化的内容
        self.prism_euler = np.zeros((3, 1))
        self.prism_translation = np.zeros((3, 1))
        # PMMA板位置坐标变化的内容
        self.PMMA_euler = np.zeros((3, 1))
        self.PMMA_translation = np.zeros((3, 1))

    def to_dict(self):
        return {
            'ccd_pixel_size': self.ccd_size_pixel.tolist(),
            'pixel_size': self.single_pixel_size.tolist(),
            'dx': self.dx,
            'dy': self.dy,
            'alpha_deg': self.alpha_deg,
            'cam_focal_length': float(self.cam_focal_length),
            'intrinsic_matrix': self.intrinsic_matrix.tolist(),
            'd': self.prism_d,
            # 'L_ol': self.L_ol,
            'L_prism_PMMA': self.L_prism_PMMA,
            'bi_prism_half_width': self.bi_prism_half_width,
            'bi_prism_half_height': self.bi_prism_half_height,
            'PMMA_thickness': self.PMMA_thickness,
            'PDMS_thickness': self.PDMS_thickness,
            'n_air': self.n_air,
            'n_prism': self.n_prism,
            'n_PMMA': self.n_PMMA,
            'n_PDMS': self.n_PDMS,
            'prism_euler': self.prism_euler.tolist(),
            'prism_translation': self.prism_translation.tolist(),
            'PMMA_euler': self.PMMA_euler.tolist(),
            'PMMA_translation': self.PMMA_translation.tolist()
        }

    def from_dict(self, dict_data):
        self.ccd_size_pixel = np.array(dict_data['ccd_pixel_size'])
        self.single_pixel_size = np.array(dict_data['pixel_size'])
        self.dx = dict_data['dx']
        self.dy = dict_data['dy']
        self.alpha_deg = dict_data['alpha_deg']
        self.cam_focal_length = dict_data['cam_focal_length']
        self.intrinsic_matrix = np.array(dict_data['intrinsic_matrix'])
        self.prism_d = dict_data['d']
        # self.L_ol = dict_data['L_ol']
        self.L_prism_PMMA = dict_data['L_prism_PMMA']
        self.bi_prism_half_width = dict_data['bi_prism_half_width']
        self.bi_prism_half_height = dict_data['bi_prism_half_height']
        self.PMMA_thickness = dict_data['PMMA_thickness']
        self.PDMS_thickness = dict_data['PDMS_thickness']
        self.n_air = dict_data['n_air']
        self.n_prism = dict_data['n_prism']
        self.n_PMMA = dict_data['n_PMMA']
        self.n_PDMS = dict_data['n_PDMS']
        self.prism_euler = np.array(dict_data['prism_euler'])
        self.prism_translation = np.array(dict_data['prism_translation'])
        self.PMMA_euler = np.array(dict_data['PMMA_euler'])
        self.PMMA_translation = np.array(dict_data['PMMA_euler'])


class BiPrismRayTracerTrans:
    """
    用于双棱镜光线追迹的理论模型
    """

    def __init__(self):
        # 后续的做法，采用通用代码处理光线追踪
        self.BP = BP()
        self.BFP = BFP()
        self.PMMA = PMMA()
        self.OptEl_sequence: list[OptElement] = []
        self.params = CameraParams()

        # 非系统结构参数
        self.left_virtual_cam_position = [0, 0, 0]
        self.right_virtual_cam_position = [0, 0, 0]
        self.ray_path_points = np.zeros((7, 3, 0))
        self.calc_result_points = np.zeros((3, 0))
        # 其他参数
        self.is_optimizing = False      # 表示是否正在优化，优化的话不会弹出任何调试内容

    def optimize(self):
        self.is_optimizing = True

    def evaluate(self):
        self.is_optimizing = False

    def save_model_params(self, filename):
        """
        保存 BP, BFP, PMMA 的参数到 JSON 文件
        :param filename: 保存的文件路径
        """
        try:
            # 构建保存的数据，直接调用各自的 save_model 方法
            model_data = {
                "BP": json.loads(self.BP.save_model()),
                "BFP": json.loads(self.BFP.save_model()),
                "PMMA": json.loads(self.PMMA.save_model()),
            }
            # 保存到 JSON 文件
            with open(filename, 'w') as file:
                json.dump(model_data, file, indent=4)

            print(f"Model parameters successfully saved to {filename}")
        except Exception as e:
            print(f"Failed to save model parameters: {e}")

    def load_model_params(self, filename):
        """
        从 JSON 文件加载 BP, BFP, PMMA 的参数
        :param filename: 加载的文件路径
        """
        if os.path.exists(filename):
            try:
                # 从 JSON 文件读取数据
                with open(filename, 'r') as file:
                    model_data = json.load(file)

                # 分别调用 BP, BFP, PMMA 的 load_model 方法加载参数
                if "BP" in model_data:
                    self.BP.load_model(json.dumps(model_data["BP"]))
                if "BFP" in model_data:
                    self.BFP.load_model(json.dumps(model_data["BFP"]))
                if "PMMA" in model_data:
                    self.PMMA.load_model(json.dumps(model_data["PMMA"]))
                print(f"Model parameters successfully loaded from {filename}")
            except Exception as e:
                print(f"Failed to load model parameters: {e}")
        else:
            raise ValueError('Model file not found!!!!!!!')

    def calc_3d_position(self, match_points):
        """
        为了提高算法运算速度，此处采用的方法是将所有的点先叠加在一起，然后统一计算三维位置坐标
        :param match_points: n*2*2 ndarray, 每个点的数据格式如下:
            [[x1, y1],
            [x2, y2]]
        :return:
        """
        # 检查输入是否为 NumPy 数组
        if not isinstance(match_points, np.ndarray):
            raise ValueError("match_points must be a NumPy ndarray.")

        # 检查输入数组的形状是否符合要求
        if len(match_points.shape) != 3 or match_points.shape[1] != 2 or match_points.shape[2] != 2:
            raise ValueError("match_points must have shape (n, 2, 2). "
                             f"Current shape is {match_points.shape}.")

        point_left = match_points[:, 0, :].T
        point_right = match_points[:, 1, :].T

        # self.clear_all_ray_points()

        p3_1, n4_1 = self.trace_ray(point_left, side=PrismSide.side_right)
        p3_2, n4_2 = self.trace_ray(point_right, side=PrismSide.side_left)

        self.calc_result_points, diss = calculate_line_line_intersection_or_closest_point(
            p3_1, n4_1, p3_2, n4_2)
        self.calc_result_points = self.calc_result_points.T
        return self.calc_result_points, diss

    def calculate_max_fov_param(self, max_end_plane_range=float("inf")):
        """
        计算在对应的最大视场的位置的z坐标
        :return:
        ret1: 对应的 z 深度
        ret2: 对应的视场范围
        """

        # 计算对应的棱镜边缘对应的成像ccd的像素坐标
        half_width_2_ccd_pixel = self.params.bi_prism_half_width / (
                self.params.prism_translation[2] + self.params.prism_d) * self.params.cam_focal_length / \
                                 self.params.single_pixel_size[0]

        p3_1, n4_1 = self.trace_ray(np.array([half_width_2_ccd_pixel, 0]), side=PrismSide.side_left)
        # p3_2, n3_2 = tracer.trace_ray(np.array([1, 0]))
        # p3_3, n3_3 = tracer.trace_ray(np.array([-0.5 * tracer.params.ccd_pixel_size[0], 0]))
        p3_4, n4_4 = self.trace_ray(np.array([-1, 0]), side=PrismSide.side_right)
        pos_1, _ = calculate_line_line_intersection_or_closest_point(p3_1, n4_1, p3_4, n4_4)
        # 记得清理这些中间变量
        self.clear_all_ray_points()

        fov_range = np.abs(pos_1[0][0] * 2)

        # 如果交点坐标太远了，那么设置最远距离
        min_z_fov = self.params.prism_translation[2] + self.params.prism_d
        # min_z_fov = self.params.L_ol + self.params.d
        max_z_range = min_z_fov + max_end_plane_range
        if pos_1[2][0] < max_z_range:
            if pos_1[2][0] < min_z_fov:
                z_fov = min_z_fov
            else:
                z_fov = pos_1[2][0]
        else:
            z_fov = max_z_range
            pos_2, _ = calculate_line_line_intersection_or_closest_point(
                p3_4, n4_4, np.array([0, 0, max_z_range]).reshape((3, 1)), np.array([1, 0, 0]).reshape((3, 1)))
            fov_range = np.abs(pos_2[0][0]) * 2

        return z_fov, fov_range

    def calculate_dof(self, L, F):
        """
        计算对应物距下的前后景深范围
        :param L:
        :return:
        """
        sigma = 0.033
        # F = 4                   # F = f/D, 光圈大小
        f = self.params.cam_focal_length
        front_dof = F * sigma * L * (L - f) / (f ** 2 + F * sigma * (L - f))
        back_dof = F * sigma * L * (L - f) / (f ** 2 - F * sigma * (L - f))

        return np.abs(front_dof), np.abs(back_dof)

    def print_params(self):
        """
        控制台打印出系统的可调结构参数。
        :return:
        """
        parameters = vars(self.params)
        for key, value in parameters.items():
            print(f"{key}: {value}")

    def clear_all_ray_points(self, print_msg=True):
        """
        清空所有记录过的光线路径点
        :return:
        """
        if print_msg is True:
            print('all points cleared!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        self.ray_path_points = np.zeros((7, 3, 0))

    def pixel_coordinate_to_world_coordinate(self, pl_pcs):
        """
        将像素坐标系坐标转换为对应的世界坐标系的坐标，并进行畸变矫正
        :param pl_pcs: 像素坐标，形状为 (2, N)，N是像素点的数量
        :return: 对应的世界坐标系坐标
        """
        # 获取相机内参和畸变系数
        intrinsic_matrix = self.params.intrinsic_matrix
        dist_coeffs = self.params.distortion_coefficients

        # 图像的尺寸（假设这是CCD尺寸）
        width, height = self.params.ccd_size_pixel

        # --------------------------------------------------
        # 1. 将像素坐标系的坐标进行中心化处理
        # 由于原始图像的像素坐标以左上角为原点，所以我们需要将其转换为相机坐标系
        pl_pcs_centered = pl_pcs + np.array([width / 2, height / 2]).reshape(2, -1)

        # --------------------------------------------------
        # 2. 使用cv2.undistortPoints去畸变
        # 将像素坐标转换为归一化相机坐标
        undistorted_points = cv2.undistortPoints(pl_pcs_centered.T, intrinsic_matrix, dist_coeffs)

        # --------------------------------------------------
        # 3. 将去畸变后的归一化坐标转换为世界坐标
        # 归一化坐标转换为世界坐标，假设Z轴值为-1
        u, v = undistorted_points[:, 0, 0], undistorted_points[:, 0, 1]
        pl0_wcs = np.stack([u, v, -np.ones_like(u)], axis=0)

        return pl0_wcs.reshape((3, -1))

    # def pixel_coordinate_to_world_coordinate(self, pl_pcs):
    #     """
    #     将像素坐标系坐标转换为对应的世界坐标系的坐标
    #     :param pl_pcs:
    #     :return:
    #     """
    #     # --------------------------------------------------
    #     # 采用归一化坐标进行计算
    #     intrinsic_matrix = self.params.intrinsic_matrix
    #
    #     u = (pl_pcs[0] + 0.5 * self.params.ccd_size_pixel[0] - intrinsic_matrix[0, 2]) / intrinsic_matrix[0, 0]
    #     v = (pl_pcs[1] + 0.5 * self.params.ccd_size_pixel[1] - intrinsic_matrix[1, 2]) / intrinsic_matrix[1, 1]
    #     pl0_wcs = np.stack([u, v, -np.ones_like(u)], axis=0)
    #
    #     return pl0_wcs.reshape((3, -1))

    def calculate_virtual_camera_position(self):
        """
        计算并返回对应的系统可调结构参数情境下的虚拟相机的位置
        :return:
        """
        self.clear_all_ray_points()
        p3_1, n3_1 = self.trace_ray(
            np.array([0.3 * self.params.ccd_size_pixel[0], 0]),
            side=PrismSide.side_left)
        p3_2, n3_2 = self.trace_ray(
            np.array([1, 0]),
            side=PrismSide.side_left)
        pos_1, _ = calculate_line_line_intersection_or_closest_point(p3_1, n3_1, p3_2, n3_2)
        p3_3, n3_3 = self.trace_ray(
            np.array([-0.3 * self.params.ccd_size_pixel[0], 0]),
            side=PrismSide.side_right)
        p3_4, n3_4 = self.trace_ray(
            np.array([-1, 0]),
            side=PrismSide.side_right)
        pos_2, _ = calculate_line_line_intersection_or_closest_point(p3_3, n3_3, p3_4, n3_4)

        # try:
        #     self.clear_all_ray_points()
        #     p3_1, n4_1 = self.trace_ray(np.array([0.5 * self.params.ccd_pixel_size[0], 0]))
        #     p3_2, n4_2 = self.trace_ray(np.array([1, 0]))
        #     pos_1, _ = calculate_line_line_intersection_or_closest_point(p3_1, n4_1, p3_2, n4_2)
        #     p3_3, n4_3 = self.trace_ray(np.array([-0.5 * self.params.ccd_pixel_size[0], 0]))
        #     p3_4, n4_4 = self.trace_ray(np.array([-1, 0]))
        #     pos_2, _ = calculate_line_line_intersection_or_closest_point(p3_3, n4_3, p3_4, n4_4)
        # except Exception as e:
        #     pos_1 = np.array([0, 0, 0])
        #     pos_2 = np.array([0, 0, 0])
        #     print('光线追踪失败！！', e)

        self.left_virtual_cam_position = pos_1
        self.right_virtual_cam_position = pos_2
        return self.left_virtual_cam_position, self.right_virtual_cam_position

    def trace_ray(self, pl_pcs, side):
        """
        从像素点出发追迹对应的光线，最后会返回对应的光线在末端的交点和方向向量
        :param pl_pcs: 出发像素点坐标
        :param side: 表示是哪个平面的
        :return:
        """
        # 1. 首先把像素坐标系的坐标值转换到世界空间
        pl0_wcs = self.pixel_coordinate_to_world_coordinate(pl_pcs.reshape((2, -1)))
        n1_wcs = -pl0_wcs

        # points_list = [pl0_wcs, np.zeros_like(pl0_wcs)]
        #
        # p_tmp, n_tmp = pl0_wcs, n1_wcs
        # for OptEl in self.OptEl_sequence:
        #     p_tmp, n_tmp, p1_wcs, n2_wcs = OptEl.trace_ray(
        #         p_tmp, n_tmp, params=side)
        #     points_list.append(p1_wcs)
        #     points_list.append(p_tmp)
        # # 为了可视化计算出射光线的点
        # pl5_wcs = calculate_line_plane_intersection_point(
        #     p_tmp, n_tmp, 0, 0, 1, -(p_tmp[2, 0] + 10))
        # points_list.append(pl5_wcs)
        # # 返回对应的三维点坐标
        # # warnings.warn('计算的光线点每次都会记录，如果绘图的话记得及时调用 clear_all_ray_points 清空三维点缓存！')
        # self.ray_path_points = np.concatenate(
        #     [self.ray_path_points,
        #      np.array(points_list)],
        #     axis=2
        # )

        pl2_wcs, n3_wcs, pl1_wcs, n2_wcs = self.BFP.trace_ray(
            pl0_wcs, n1_wcs, params=side)

        pl4_wcs, n5_wcs, pl3_wcs, n4_wcs = self.PMMA.trace_ray(
            pl2_wcs, n3_wcs)

        # 为了可视化计算出射光线的点
        pl5_wcs = calculate_line_plane_intersection_point(
            pl4_wcs, n5_wcs, 0, 0, 1, -(pl4_wcs[2, 0] + 10))

        # 返回对应的三维点坐标
        # warnings.warn('计算的光线点每次都会记录，如果绘图的话记得及时调用 clear_all_ray_points 清空三维点缓存！')
        if self.is_optimizing is False:
            self.ray_path_points = np.concatenate(
                [self.ray_path_points,
                 np.array([pl0_wcs, np.zeros_like(pl0_wcs), pl1_wcs, pl2_wcs, pl3_wcs, pl4_wcs, pl5_wcs])],
                axis=2
            )

        return pl4_wcs, n5_wcs

    def plot_3d_scene(self, ax=None, plot_rays=False, plot_result_points=False):
        """
        可视化对应的系统结构及光线路径，虚拟相机位置
        :param plot_result_points:
        :param plot_rays:
        :param ax:
        :return:
        """
        tim_start = time.time()
        if ax is None:
            # 创建画布
            fig = plt.figure()
            # 创建 3D 坐标系
            ax = fig.add_subplot(111, projection='3d')

        # ----------------------------------------------------------------------
        # ccd 绘制
        ccd_half_real_size = self.params.pixel_size * self.params.ccd_size_pixel[0] * 0.5
        x = np.linspace(-ccd_half_real_size[0], ccd_half_real_size[0], 2)
        y = np.linspace(-ccd_half_real_size[1], ccd_half_real_size[1], 2)
        X, Y = np.meshgrid(x, y)
        # 平面 z=4.5 的部分
        ax.plot_surface(X, Y, Z=-np.ones_like(X), color='g', alpha=0.6)

        self.BFP.plot_OptEl(ax=ax)
        self.PMMA.plot_OptEl(ax=ax)

        # ------------------------------------------------------------------
        if plot_rays:
            # 绘制所有光线
            for i in range(self.ray_path_points.shape[2]):
                x = self.ray_path_points[:, 0, i]
                y = self.ray_path_points[:, 1, i]
                z = self.ray_path_points[:, 2, i]
                ax.plot(x, y, z, 'b', linewidth=0.5)  # 直接绘制光线路径
                ax.scatter(x, y, z, c='r', marker='o', s=7)

        # 绘制最终的数据点
        if plot_result_points and self.calc_result_points.shape[1] != 0:
            ax.scatter(self.calc_result_points[:, 0],
                       self.calc_result_points[:, 1],
                       self.calc_result_points[:, 2],
                       c='g', marker='o', s=7)

        ax.set(xlabel='X', ylabel='Y', zlabel='Z')

        # ---------------------------------------------------------------------
        # 虚拟双目位置绘制
        l_p = self.left_virtual_cam_position
        r_p = self.right_virtual_cam_position
        ax.scatter(l_p[0], l_p[1], l_p[2], c='g', marker='*')
        ax.scatter(r_p[0], r_p[1], r_p[2], c='g', marker='*')
        ax.set_box_aspect([1, 1, 1])  # x:y:z = 1:1:1
        # 在指定位置添加文本标签
        # ax.text3D(l_p[0], l_p[1], l_p[2], 'l_cam', fontsize=12, color='red')
        # ax.text3D(r_p[0], r_p[1], r_p[2], 'r_cam', fontsize=12, color='red')
        if self.is_optimizing:
            print('left camera position: ', l_p)
            print('right camera position: ', r_p)
            print('time cost: {}s'.format(time.time() - tim_start))

        return ax


def get_latest_tracer(calibration_npz_path='../npy_files/camera_calibration.npz'):
    """
    参数一直在变化，这个是最新的 tracer 参数
    :return:
    """
    # 导入必要参数
    tracer = BiPrismRayTracerTrans()
    camera_params = BiPrismCameraCalibration()
    # 设置合理初始值
    camera_params.load_camera_params(
        calibration_result_file_name=calibration_npz_path)
    tracer.params = camera_params.camera_params

    # 此处的相对位置关系正确
    tracer.BFP.set_world_to_OptEl_rotate_translation_matrix(
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]),
        translate_matrix=np.array([[0], [0], [-27.9]])      # -25.4073877845036
    )
    tracer.BFP.n2 = 1.49
    tracer.PMMA.set_world_to_OptEl_rotate_translation_matrix(
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        translate_matrix=np.array([[0], [0], [-53]])
    )
    tracer.PMMA.n2 = 1.49
    tracer.PMMA.n3 = 1
    return tracer

def visualMode_tracer(model_path,calibration_npz_path='../npy_files/camera_matrix_435.npz'):
    """
        参数一直在变化，这个是最新的 tracer 参数
        :return:
        """
    # 导入必要参数
    tracer = BiPrismRayTracerTrans()
    tracer.load_model_params(model_path)
    # tracer.BP.load_model(json.dumps(model_data['BP']))
    # tracer.BFP.load_model(json.dumps(model_data['BFP']))
    # tracer.PMMA.load_model(json.dumps(model_data['PMMA']))
    camera_params = BiPrismCameraCalibration()
    # # 设置合理初始值
    camera_params.load_camera_params(
        calibration_result_file_name=calibration_npz_path)
    tracer.params = camera_params.camera_params
    # tracer.BFP.set_world_to_OptEl_rotate_translation_matrix(
    #     np.array([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, -1]
    #     ]),
    #     translate_matrix=np.array([[0], [0], [-27.794498299901537]])  # -25.4073877845036
    # )
    # tracer.BFP.n2 = 1.5
    # tracer.PMMA.set_world_to_OptEl_rotate_translation_matrix(
    #     np.array([
    #         [1, 0, 0],
    #         [0, 1, 0],
    #         [0, 0, 1]
    #     ]),
    #     translate_matrix=np.array([[0], [0], [-53]])
    # )
    # tracer.PMMA.n3 = 1
    #
    # # 此处的相对位置关系正确
    # tracer.BFP.set_world_to_OptEl_rotate_translation_matrix(
    #     np.array([
    #         [
    #             0.9993206826350703,
    #             0.01000725070060006,
    #             -0.035468693113679034
    #         ],
    #         [
    #             -0.010491132295098023,
    #             0.9998540652749813,
    #             -0.013482740680594251
    #         ],
    #         [
    #             -0.035328591833580836,
    #             -0.013845688372512918,
    #             -0.9992798344370547
    #         ]
    #     ]),
    #     translate_matrix=np.array([[
    #             -0.2402680765407833
    #         ],
    #         [
    #             -0.4086206455178209
    #         ],
    #         [
    #             -27.69164250514755
    #         ]])  # -25.4073877845036
    # )
    # tracer.BFP.n2 = 1.49
    # tracer.PMMA.set_world_to_OptEl_rotate_translation_matrix(
    #     np.array([
    #         [
    #             0.9974842296998393,
    #             0.002859324166764852,
    #             0.07083103673833746
    #         ],
    #         [
    #             0.00285932416676483,
    #             0.9967502062131336,
    #             -0.08050373084140254
    #         ],
    #         [
    #             -0.07083103673833746,
    #             0.08050373084140254,
    #             0.9942344359129729
    #         ]
    #     ]),
    #     translate_matrix=np.array([[0], [0], [-51]])
    # )
    # tracer.PMMA.n3 = 1.45

    return tracer
def check_bfp_shape():
    # 创建 BFP 实例
    bfp = BFP()

    # 设置一个旋转和平移矩阵
    rotate_matrix = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    # translate_matrix = np.array([[0], [0], [-25.4073877845036]])
    translate_matrix = np.array([[0], [0], [0]])
    bfp.set_world_to_OptEl_rotate_translation_matrix(rotate_matrix, translate_matrix)

    # 定义 -11 到 11 的 x, y 区域
    x_count, y_count = 20, 20
    x = np.linspace(-11, 11, x_count)  # x 方向 100 个点
    y = np.linspace(-11, 11, y_count)  # y 方向 100 个点
    x, y = np.meshgrid(x, y)  # 创建网格

    # 计算对应的 z 值
    # z = np.ones_like(x) * 10
    z = bfp.full_surface_fun(x, y)
    # 转换为 10000x3 的 ndarray
    xyz_flat = np.vstack((x.ravel(), y.ravel(), z.ravel()))

    xyz_flat_transformed = bfp.world_coordinate_to_OptEl_coordinate(xyz_flat, False)

    # 恢复为原始的 100x100 的 x, y, z
    x_rec = xyz_flat_transformed[0, :].reshape(x_count, y_count)
    y_rec = xyz_flat_transformed[1, :].reshape(x_count, y_count)
    z_rec = xyz_flat_transformed[2, :].reshape(x_count, y_count)

    # 绘制 3D 图
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x_rec, y_rec, z_rec, cmap='viridis', edgecolor='none')
    ax.set_title("BFP Left Surface Shape")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    plt.axis('equal')
    plt.show()


def check_bfp_in_sys():
    # 导入必要参数
    tracer = BiPrismRayTracerTrans()
    camera_params = BiPrismCameraCalibration()
    # 设置合理初始值
    camera_params.load_camera_params(
        calibration_result_file_name='../data/paper data/camera calibration/camera_calibration.npz')
    tracer.params = camera_params.camera_params

    # 此处的相对位置关系正确
    tracer.BFP.set_world_to_OptEl_rotate_translation_matrix(
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]),
        translate_matrix=np.array([[0], [00], [-28]])      # -25.4073877845036
    )
    # 0.01, 0.90, -25.87
    tracer.BFP.n2 = 1.49
    tracer.PMMA.set_world_to_OptEl_rotate_translation_matrix(
        np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),
        translate_matrix=np.array([[0], [0], [-58]])
    )

    tracer.PMMA.n2 = 1.45
    # tracer.PMMA.n3 = 1
    tracer.OptEl_sequence.append(tracer.BFP)
    tracer.OptEl_sequence.append(tracer.PMMA)
    # points_data1 = np.transpose(
    #     np.load('../data/BFP Tac AF1023/MCP/captured_image_20241227_200113_point_pairs.npy'),
    #     (1, 0, 2))
    # points_data2 = np.transpose(
    #     np.load('../data/BFP Tac AF1023/MCP/captured_image_20241227_200127_point_pairs.npy'),
    #     (1, 0, 2))

    points_data1 = np.load('../data/paper data/MCD/0.0000_0.0000_0.0000_markers_1.npy')
    points_data2 = np.load('../data/paper data/MCD/0.0000_0.0000_0.1000_markers_1.npy')

    # # 定义你想减去的两个数，例如 [a, b]
    # subtract_values = np.array([[1920 * 0.5, 1080 * 0.5],[1920 * 0.5, 1080 * 0.5]]).astype(np.int16)  # 替换为你具体的值
    # points_data1 = points_data1 - subtract_values[np.newaxis, :]
    # points_data2 = points_data2 - subtract_values[np.newaxis, :]

    # points_data = np.load('../test/new_pairs.npy')
    # test_data = points_data[0, :, :][np.newaxis, :, :]
    # p1_3d, _ = tracer.calc_3d_position(test_data)
    # test_data[:, 0, 0] += 1
    # p2_3d, _ = tracer.calc_3d_position(test_data)

    time_start = time.time()
    p1, _ = tracer.calc_3d_position(points_data1)
    p2, _ = tracer.calc_3d_position(points_data2)

    print(time.time()-time_start)
    result = copy.deepcopy(p2)
    result[:, 2] = (p2-p1)[:, 2]
    plot_real_points_and_calc_points(result, plt_show=False)
    # plot_real_points_and_calc_points(p1)
    # tracer.calculate_virtual_camera_position()
    tracer.plot_3d_scene()
    plt.axis('equal')
    plt.show()


# 更新主函数测试部分
if __name__ == '__main__':
    # check_bfp_in_sys()
    tracer=visualMode_tracer()


    # # 定义你想减去的两个数，例如 [a, b]
    # subtract_values = np.array([[1920 * 0.5, 1080 * 0.5],[1920 * 0.5, 1080 * 0.5]]).astype(np.int16)  # 替换为你具体的值
    # points_data1 = points_data1 - subtract_values[np.newaxis, :]
    # points_data2 = points_data2 - subtract_values[np.newaxis, :]

    # points_data = np.load('../test/new_pairs.npy')
    # test_data = points_data[0, :, :][np.newaxis, :, :]
    # p1_3d, _ = tracer.calc_3d_position(test_data)
    # test_data[:, 0, 0] += 1
    # p2_3d, _ = tracer.calc_3d_position(test_data)
#     points=np.array([
#  [[-657.5, 274.5], [549.5, 265.5]],
#  [[-657.5, 311.5], [543.5, 303.5]],
#  [[-577.0, 346.0], [616.0, 340.0]]
# ])
    points = np.array([[[-625.42737, -35.04776],
                        [617.3037, -45.770844]],
                       [[-625.1776, -36.568268],
                        [617.08984, -47.252625]],
                       [[-571.35864, - 36.07132],
                        [672.7964, - 46.783478]]
                       ])
    p1, d = tracer.calc_3d_position(points)
    print(p1)
    print(d)
    tracer.plot_3d_scene(plot_rays=True,plot_result_points=True)
    plt.axis('equal')
    plt.show()