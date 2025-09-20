import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- 2D版本 ---
def calculate_ray_line_intersections(ray_origins, ray_directions, line_normal, line_point):
    """
    计算2D光线与一条无限长直线的交点。

    直线由法线向量(line_normal)和直线上的一点(line_point)定义。

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


class PlanarLossEvaluator:
    """
    【2D版本】评估光线与一个有限大小的目标线段的交互，并计算损失函数。
    """
    def __init__(self, line_normal, line_center, line_length, weights, grid_size=20):
        self.normal = np.array(line_normal, dtype=float) / np.linalg.norm(line_normal)
        self.center = np.array(line_center, dtype=float).reshape(2, 1)
        self.length = float(line_length)
        self.weights = weights
        self.grid_size = grid_size
        
        # 预计算线段上的局部坐标系基向量 (u_axis)，即线段的切线方向
        # 通过将法线旋转90度得到: (nx, ny) -> (-ny, nx)
        self.u_axis = np.array([-self.normal[1], self.normal[0]])

    def _calculate_parallelism_loss(self, exit_dirs):
        """
        计算平行度损失 (光线与光线之间)。此函数逻辑与维度无关。
        """
        if exit_dirs.shape[1] < 2:
            return 1.0

        mean_vector = np.mean(exit_dirs, axis=1)
        magnitude = np.linalg.norm(mean_vector)
        return 1.0 - magnitude

    def _calculate_uniformity_coverage_loss(self, local_points_u):
        """
        【2D版本】使用1D直方图计算均匀度和覆盖度损失。
        :param local_points_u: np.ndarray, 光线在线段局部坐标系下的u坐标，形状 (K,)
        """
        if local_points_u.shape[0] == 0:
            return 1.0, 1.0

        # 沿线段创建1D的统计“箱子”
        bins = np.linspace(-self.length / 2, self.length / 2, self.grid_size + 1)
        
        # 使用1D直方图统计每个箱子里的光线数量
        flux_map, _ = np.histogram(local_points_u, bins=bins)
        
        illuminated_cells = np.count_nonzero(flux_map)
        total_cells = self.grid_size
        coverage = illuminated_cells / total_cells
        coverage_loss = 1.0 - coverage

        if illuminated_cells == 0:
            return 1.0, coverage_loss

        flux_in_lit_cells = flux_map[flux_map > 0]
        mean_flux = np.mean(flux_in_lit_cells)
        std_dev = np.std(flux_in_lit_cells)
        uniformity_loss = std_dev / mean_flux if mean_flux > 0 else 1.0
        return uniformity_loss, coverage_loss

    def evaluate(self, ray_origins, ray_directions, quiet=False):
        """
        【2D版本】评估光线并计算总损失。
        """
        intersection_points, valid_mask = calculate_ray_line_intersections(
            ray_origins, ray_directions, self.normal, self.center.flatten()
        )
        
        if intersection_points.shape[1] == 0:
            loss_p, loss_u, loss_c = 1.0, 1.0, 1.0
            final_loss = sum(self.weights)
            if not quiet:
                print(f"Loss -> Total: {final_loss:.4f} | Parallelism: {loss_p:.4f} | Uniformity: {loss_u:.4f} | Coverage: {loss_c:.4f} (No intersections)")
            return final_loss, (loss_p, loss_u, loss_c), np.array([]).reshape(2, 0)

        # 筛选落在目标线段长度范围内的交点
        relative_vectors = intersection_points - self.center
        u_coords = np.dot(relative_vectors.T, self.u_axis)
        boundary_mask = np.abs(u_coords) <= self.length / 2
        
        final_points_world_2d = intersection_points[:, boundary_mask]
        final_points_local_1d = u_coords[boundary_mask]
        final_dirs = ray_directions[:, valid_mask][:, boundary_mask]
        
        # 计算各项损失
        loss_p = self._calculate_parallelism_loss(final_dirs)
        loss_u, loss_c = self._calculate_uniformity_coverage_loss(final_points_local_1d)
        
        # 加权计算总损失
        w_p, w_u, w_c = self.weights
        final_loss = w_p * loss_p + w_u * loss_u + w_c * loss_c
        
        if not quiet:
            print(f"Loss -> Total: {final_loss:.4f} | Parallelism: {loss_p:.4f} | Uniformity: {loss_u:.4f} | Coverage: {loss_c:.4f} | Rays Hit: {final_dirs.shape[1]}/{ray_origins.shape[1]}")
        
        return final_loss, (loss_p, loss_u, loss_c), final_points_world_2d

    def plot_element_2d(self, ax, color='cyan', **kwargs):
        """
        【2D版本】在给定的 Matplotlib 2D 坐标轴上绘制目标线段。
        """
        # 计算线段的两个端点
        p1 = self.center - (self.length / 2.0) * self.u_axis.reshape(2, 1)
        p2 = self.center + (self.length / 2.0) * self.u_axis.reshape(2, 1)
        ax.plot([p1[0,0], p2[0,0]], [p1[1,0], p2[1,0]], color=color,  label='评估平面 (Target)', **kwargs)

    def save_to_image(self, filename, intersection_points, image_size=(512, 64), 
                      colormap=plt.cm.hot):
        """
        【2D版本】将线段上的光线分布保存为一张热力图。
        
        :param filename: str, 保存的文件名, e.g., 'output.png'
        :param intersection_points: np.ndarray, 二维交点坐标 (2, K)
        :param image_size: tuple, 输出图片的尺寸 (宽, 高)
        :param colormap: matplotlib colormap, 用于表示强度的色条
        """
        width, height = image_size
        image = np.zeros((height, width, 3), dtype=np.uint8)

        if intersection_points.shape[1] == 0:
            print("Warning: No intersection points to save in the image.")
            Image.fromarray(image, 'RGB').save(filename)
            return

        # 1. 将世界坐标交点转换为局部u坐标
        relative_vectors = intersection_points - self.center
        u_coords = np.dot(relative_vectors.T, self.u_axis)
        
        # 2. 使用1D直方图计算能量分布
        bins = np.linspace(-self.length / 2, self.length / 2, width + 1)
        flux_map, _ = np.histogram(u_coords, bins=bins)
        
        # 3. 将能量分布（光子通量）映射为颜色
        if np.max(flux_map) > 0:
            # 归一化到 [0, 1]
            normalized_flux = flux_map / np.max(flux_map)
            # 应用色彩映射（注意 colormap 返回的是 RGBA）
            colors = colormap(normalized_flux)[:, :3] * 255
        else:
            colors = np.zeros((width, 3))
            
        # 4. 在图像上绘制垂直色带
        for i in range(width):
            image[:, i, :] = colors[i]

        # 5. 保存图片
        Image.fromarray(image, 'RGB').save(filename)
        print(f"Energy distribution image saved to {filename}")

# --- 示例用法 ---
if __name__ == '__main__':
    # 1. 创建一个评估器实例：位于y=20，长度为30的水平线段
    evaluator = PlanarLossEvaluator(
        line_normal=[0, 1], 
        line_center=[0, 20], 
        line_length=30, 
        weights=[0.3, 0.4, 0.3]
    )

    # 2. 模拟一些光线数据
    num_rays = 50000
    # 假设光线从y=0发出，向评估器汇聚，但带有随机散射
    ray_origins_2d = np.zeros((2, num_rays))
    directions = np.array([0, 20])[:, np.newaxis] - ray_origins_2d
    directions /= np.linalg.norm(directions, axis=0)
    # 添加随机性
    noise = (np.random.rand(2, num_rays) - 0.5) * 0.8
    ray_directions_2d = directions + noise
    ray_directions_2d /= np.linalg.norm(ray_directions_2d, axis=0)

    # 3. 执行评估
    loss, components, final_points = evaluator.evaluate(ray_origins_2d, ray_directions_2d)

    # 4. 可视化评估平面和交点
    fig, ax = plt.subplots(figsize=(10, 8))
    evaluator.plot_OptEl(ax)
    if final_points.shape[1] > 0:
        ax.scatter(final_points[0, :], final_points[1, :], s=1, c='red', alpha=0.5, label=f'{final_points.shape[1]} Hits')
    
    ax.set_title("2D Loss Evaluator Visualization")
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect('equal', 'box')
    ax.legend()
    ax.grid(True)
    plt.show()

    # 5. 保存光斑分布图
    evaluator.save_to_image("spot_distribution_2d.png", final_points)