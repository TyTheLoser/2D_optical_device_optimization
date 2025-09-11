import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
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
class PlanarLossEvaluator:
    """
    评估光线与一个有限大小的目标平面的交互，并计算损失函数。
    """
    def __init__(self, plane_normal, plane_center, plane_width, plane_height, weights, grid_size=20):
        # (构造函数与之前版本相同)
        self.normal = np.array(plane_normal, dtype=float) / np.linalg.norm(plane_normal)
        self.center = np.array(plane_center, dtype=float).reshape(3, 1)
        self.width = float(plane_width)
        self.height = float(plane_height)
        self.weights = weights
        self.grid_size = grid_size
        
        # 预计算平面上的局部坐标系基向量 (u, v)
        if np.allclose(np.abs(self.normal), [0, 0, 1]):
            self.u_axis = np.array([1, 0, 0])
        else:
            self.u_axis = np.cross([0, 0, 1], self.normal)
            self.u_axis /= np.linalg.norm(self.u_axis)
        self.v_axis = np.cross(self.normal, self.u_axis)
        self.v_axis /= np.linalg.norm(self.v_axis)

    # ----------------------------------------------------------------------------
    # 2. 已修正的平行度损失函数
    # ----------------------------------------------------------------------------
    def _calculate_parallelism_loss(self, exit_dirs):
        """
        计算平行度损失 (光线与光线之间)。

        通过计算所有方向向量的平均向量来实现。如果所有光线完美平行，
        平均向量的长度将为1。光线越发散，平均向量长度越接近0。
        损失 = 1 - |mean_vector|
        """
        if exit_dirs.shape[1] < 2:
            return 1.0  # 如果光线太少，施加惩罚

        # 计算所有方向向量的平均向量
        mean_vector = np.mean(exit_dirs, axis=1)
        
        # 计算平均向量的模长（长度）
        magnitude = np.linalg.norm(mean_vector)
        
        # 模长越接近1，说明方向越一致，损失越小
        return 1.0 - magnitude

    def _calculate_uniformity_coverage_loss(self, local_points):
        # (此函数与之前版本相同)
        if local_points.shape[1] == 0:
            return 1.0, 1.0
        u_bins = np.linspace(-self.width / 2, self.width / 2, self.grid_size + 1)
        v_bins = np.linspace(-self.height / 2, self.height / 2, self.grid_size + 1)
        flux_map, _, _ = np.histogram2d(local_points[0, :], local_points[1, :], bins=[u_bins, v_bins])
        illuminated_cells = np.count_nonzero(flux_map)
        total_cells = self.grid_size * self.grid_size
        coverage = illuminated_cells / total_cells
        coverage_loss = 1.0 - coverage
        if illuminated_cells == 0:
            return 1.0, coverage_loss
        flux_in_lit_cells = flux_map[flux_map > 0]
        mean_flux = np.mean(flux_in_lit_cells)
        std_dev = np.std(flux_in_lit_cells)
        uniformity_loss = std_dev / mean_flux if mean_flux > 0 else 1.0
        return uniformity_loss, coverage_loss

    def evaluate(self, ray_origins, ray_directions):
        # 1. 使用外部函数计算交点
        intersection_points, valid_mask = calculate_ray_plane_intersections(
            ray_origins, ray_directions, self.normal, self.center.flatten()
        )
        
        if intersection_points.shape[1] == 0:
            loss_p, loss_u, loss_c = 1.0, 1.0, 1.0 # 最大惩罚
            final_loss = sum(self.weights)
            print(f"Loss -> Total: {final_loss:.4f} | Parallelism: {loss_p:.4f} | Uniformity: {loss_u:.4f} | Coverage: {loss_c:.4f} (No intersections)")
            return final_loss, (loss_p, loss_u, loss_c), np.array([]).reshape(3,0)

        # 2. 筛选落在平面边界内的交点
        relative_vectors = intersection_points - self.center
        u_coords = np.dot(relative_vectors.T, self.u_axis)
        v_coords = np.dot(relative_vectors.T, self.v_axis)
        boundary_mask = (np.abs(u_coords) <= self.width / 2) & (np.abs(v_coords) <= self.height / 2)
        
        final_points_world_3d = intersection_points[:, boundary_mask]
        final_points_local_2d = np.vstack((u_coords[boundary_mask], v_coords[boundary_mask]))
        final_dirs = ray_directions[:, valid_mask][:, boundary_mask]
        
        # 3. 计算各项损失
        loss_p = self._calculate_parallelism_loss(final_dirs)
        loss_u, loss_c = self._calculate_uniformity_coverage_loss(final_points_local_2d)
        
        # 4. 加权计算总损失
        a, b, c = self.weights
        final_loss = a * loss_p + b * loss_u + c * loss_c
        
        print(f"Loss -> Total: {final_loss:.4f} | Parallelism: {loss_p:.4f} | Uniformity: {loss_u:.4f} | Coverage: {loss_c:.4f} | Rays Hit: {final_dirs.shape[1]}/{ray_origins.shape[1]}")
        
        # 返回总损失、损失分量元组、以及落在边界内的三维交点
        return final_loss, (loss_p, loss_u, loss_c), final_points_world_3d

    # ----------------------------------------------------------------------------
    # 3. 新增的可视化函数
    # ----------------------------------------------------------------------------
    def plot_OptEL(self, ax, plane_color='cyan'):
        """
        在给定的 Matplotlib 3D 坐标轴上绘制目标平面。

        该函数通过以下步骤工作：
        1. 在平面的局部二维(u,v)坐标系中创建一个2x2的网格。
        2. 将这个二维网格上的所有点，通过平面的中心点和基向量，映射到三维世界坐标系。
        3. 使用 plot_surface 函数绘制这个三维的矩形面片。
        """
        # 1. 在局部坐标系中定义网格的范围
        #    我们只需要2x2的网格就能定义一个矩形的四个角
        u = np.linspace(-self.width / 2, self.width / 2, 2)
        v = np.linspace(-self.height / 2, self.height / 2, 2)
        uu, vv = np.meshgrid(u, v) # 创建2x2的网格

        # 2. 将2x2的局部网格点转换为三维世界坐标
        #    公式: P_3d = Center + u_coord * U_axis + v_coord * V_axis
        #    我们利用NumPy的广播机制来高效地计算所有四个点
        #    self.center shape: (3,1)
        #    uu, vv shapes: (2,2)
        #    self.u_axis, self.v_axis shapes: (3,)
        X = self.center[0] + uu * self.u_axis[0] + vv * self.v_axis[0]
        Y = self.center[1] + uu * self.u_axis[1] + vv * self.v_axis[1]
        Z = self.center[2] + uu * self.u_axis[2] + vv * self.v_axis[2]

        # 3. 使用 plot_surface 绘制这个三维矩形平面
        ax.plot_surface(X, Y, Z, color=plane_color, alpha=0.4, shade=False, edgecolor='k', linewidth=0.5)

    def save_to_image(self, filename, intersection_points, image_size=(512, 512), 
                      point_color=(255, 0, 0), point_size=4):
        """
        将平面和交点保存为一张图片。

        :param filename: str, 保存的文件名, e.g., 'output.png'
        :param intersection_points: np.ndarray, 三维交点坐标 (3, K)
        :param image_size: tuple, 输出图片的尺寸 (宽, 高)
        :param point_color: tuple, 交点的RGB颜色 (0-255)
        :param point_size: int, 每个交点绘制的正方形像素块大小
        """
        # 创建一张黑色背景的图片
        image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)

        if intersection_points.shape[1] == 0:
            print("Warning: No intersection points to save in the image.")
            img = Image.fromarray(image, 'RGB')
            img.save(filename)
            return

        # a. 将三维世界交点转换为二维局部坐标 (u,v)
        relative_vectors = intersection_points - self.center
        u_coords = np.dot(relative_vectors.T, self.u_axis)
        v_coords = np.dot(relative_vectors.T, self.v_axis)
        
        # b. 将局部坐标 (u,v) 映射到图片的像素坐标 (px, py)
        # u -> px (0, width), v -> py (0, height)
        px = ((u_coords / self.width) + 0.5) * image_size[0]
        py = ((-v_coords / self.height) + 0.5) * image_size[1] # v轴取反以匹配图片坐标系(左上角为原点)

        # c. 在图片上绘制指定大小的像素块
        for i in range(len(px)):
            px_int, py_int = int(px[i]), int(py[i])
            
            # 确保绘制在图像边界内
            if 0 <= px_int < image_size[0] and 0 <= py_int < image_size[1]:
                x_start, y_start = px_int - point_size // 2, py_int - point_size // 2
                x_end, y_end = x_start + point_size, y_start + point_size
                
                # 再次裁剪确保不越界
                x_start, y_start = max(0, x_start), max(0, y_start)
                x_end, y_end = min(image_size[0], x_end), min(image_size[1], y_end)

                image[y_start:y_end, x_start:x_end] = point_color

        # d. 保存图片
        img = Image.fromarray(image, 'RGB')
        img.save(filename)
        print(f"Image saved to {filename}")