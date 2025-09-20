import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.collections import LineCollection
import time
from utils.utils_ray_trace import LC_device, Point_light_source
from utils.utils_lossEvaluator import PlanarLossEvaluator, calculate_ray_line_intersections
from utils.utils_free_form_prism_design import visualize_scene
from scipy.optimize import differential_evolution


# =============================================================================
# 3. 动态光源生成与优化目标封装 (2D版本)
# =============================================================================
def calculate_light_sources_from_params(a, l1, l2, l3):
    """
    【2D版本】根据斜率a和三个长度比例l1,l2,l3，计算三个光源在2D平面上的位置和姿态。

    :param a: 直线的斜率 (y = a*x + c)
    :param l1, l2, l3: 三个光源在线段上的相对位置比例 [-1, 1]
    :return: light_sources_config_2d 列表，包含2D的位置和旋转矩阵
    """
    # --- a. 计算直线与矩形的交点 (逻辑不变, z -> y) ---
    rect_bounds = {'x_min': 2.8, 'x_max': 11.66, 'y_min': -5.5, 'y_max': -1.0}
    intersections = []
    # 直线方程: y = a*x - 7.23*a - 3.25
    x_at_ymin = rect_bounds['x_min']
    y_at_ymin = a * x_at_ymin - 7.23 * a - 3.25
    if rect_bounds['y_min'] <= y_at_ymin <= rect_bounds['y_max']:
        intersections.append(np.array([x_at_ymin, y_at_ymin]))

    x_at_ymax = rect_bounds['x_max']
    y_at_ymax = a * x_at_ymax - 7.23 * a - 3.25
    if rect_bounds['y_min'] <= y_at_ymax <= rect_bounds['y_max']:
        intersections.append(np.array([x_at_ymax, y_at_ymax]))

    if a != 0:
        y_at_ymin_bound = rect_bounds['y_min']
        x_at_ymin_bound = (y_at_ymin_bound + 7.23 * a + 3.25) / a
        if rect_bounds['x_min'] <= x_at_ymin_bound <= rect_bounds['x_max']:
            intersections.append(np.array([x_at_ymin_bound, y_at_ymin_bound]))

        y_at_ymax_bound = rect_bounds['y_max']
        x_at_ymax_bound = (y_at_ymax_bound + 7.23 * a + 3.25) / a
        if rect_bounds['x_min'] <= x_at_ymax_bound <= rect_bounds['x_max']:
            intersections.append(np.array([x_at_ymax_bound, y_at_ymax_bound]))

    # 去重并确保只有两个交点
    unique_intersections = np.unique(np.round(intersections, decimals=5), axis=0)
    if len(unique_intersections) != 2:
        return []
    
    p_start, p_end = unique_intersections
    
    # --- b. 计算线段中心和半长向量 (逻辑不变) ---
    segment_center = (p_start + p_end) / 2.0
    segment_half_vector = (p_end - p_start) / 2.0

    # --- c. 计算光源位置 (简化为2D) ---
    ratios = [l1, l2, l3]
    positions_2d = [segment_center + r * segment_half_vector for r in ratios]
    # 将每个位置转换为 (2, 1) 的列向量
    positions_2d_col = [pos.reshape(2, 1) for pos in positions_2d]

    ####################################################################
    # ## d. 计算光源姿态 (2D旋转矩阵) ##
    ####################################################################
    
    # 斜率为 a 的直线，其方向向量为 [1, a]
    # 与其垂直的方向向量为 [-a, 1] (因为点积 1*(-a) + a*1 = 0)
    # 我们将这个垂直向量作为光源的“朝向”（即本地坐标系的y'轴）
    
    # 归一化这个方向向量
    norm = np.sqrt(a**2 + 1)
    # 这是新的 y' 轴方向
    new_y_axis = np.array([-a / norm, 1 / norm])
    # 新的 x' 轴必须与 y' 轴垂直，可以通过旋转90度得到
    new_x_axis = np.array([new_y_axis[1], -new_y_axis[0]]) # 即 [1/norm, a/norm]
    
    # 2D旋转矩阵的列就是新的基向量
    rotation_matrix_2d = np.array([
        [new_x_axis[0], new_y_axis[0]],
        [new_x_axis[1], new_y_axis[1]]
    ])
    
    ####################################################################
    
    # --- e. 构建2D光源配置列表 ---
    light_sources_config_2d = []
    for pos in positions_2d_col:
        light_sources_config_2d.append({
            'OptEl_to_world_translation_matrix': pos,
            "OptEl_to_world_rotation_matrix": rotation_matrix_2d
        })
        
    return light_sources_config_2d

class Objective:
    def __init__(self, device_class, light_source_class, loss_evaluator, device_static_config, constraints_config):
        """
        构造函数更新：存储静态配置，包括基准线。
        """
        self.lc_class = device_class
        self.light_source_class = light_source_class
        self.loss_evaluator = loss_evaluator
        
        # 存储静态信息，这些信息在优化过程中不变
        self.device_static_cfg = device_static_config
        
        # 参数数量由控制点数量决定
        self.num_up = self.device_static_cfg['num_up_control_points']
        self.num_down = self.device_static_cfg['num_down_control_points']

        # 其他初始化
        cfg = constraints_config
        self.x_samples = np.linspace(cfg['x_range'][0], cfg['x_range'][1], cfg.get('sample_resolution', 50))
        self.penalty_weight = cfg.get('penalty_weight', 1000.0)
        self.last_loss, self.last_penalty = float('inf'), float('inf')

    def _calculate_thickness_penalty(self, device):
        y_up, y_down = device.up_surface_fun(self.x_samples), device.down_surface_fun(self.x_samples)
        y_up_limit=40-self.x_samples
        p1 = self.penalty_weight * np.maximum(0, -np.min(y_up - y_down))**2
        p2 = self.penalty_weight * np.maximum(0, -np.min(y_down - 0))**2 # 假设下表面y>0
        p3 = self.penalty_weight * np.maximum(0, np.max(y_up - y_up_limit))**2 # 假设上表面y<40
        return p1 + p2+p3

    def __call__(self, params):
        """
        __call__ 方法重大更新：
        1. 将接收到的 'params' 解释为 'offsets' (偏移量)。
        2. 结合基准线计算最终的 'y_coords'。
        3. 使用正确的参数名 ('up_y_coords') 来创建 SplineDevice。
        """
        # 1. 将优化器传入的 params 向量正确地切片为偏移量和光源参数
        up_offsets = params[:self.num_up]
        down_offsets = params[self.num_up : self.num_up + self.num_down]
        light_params = params[self.num_up + self.num_down:]

        light_configs = calculate_light_sources_from_params(*light_params)
        if not light_configs: return 1e9
        
        # 2. 从存储的静态配置中获取基准线
        base_up_y = self.device_static_cfg['base_up_y_line']
        base_down_y = self.device_static_cfg['base_down_y_line']
        
        # 3. 计算最终的Y坐标
        final_up_y = base_up_y - up_offsets
        final_down_y = base_down_y + down_offsets

        # 4. 使用 SplineDevice 类期望的正确参数名来构建配置字典
        device_cfg = {
            'up_y_coords': final_up_y,
            'down_y_coords': final_down_y,
            'bound': self.device_static_cfg['bound'],
            'num_control_points_up': self.num_up,
            'num_control_points_down': self.num_down
        }
        device = self.lc_class(**device_cfg)
        
        # --- 后续的光线追迹和损失评估逻辑保持不变 ---
        sources = [self.light_source_class(**cfg) for cfg in light_configs]
        all_p, all_n = zip(*[s.trace_ray() for s in sources])
        p_initial, n_initial = np.hstack(all_p), np.hstack(all_n)
        
        _,_, p1, n2 = device.trace_ray(p_initial, n_initial)
        self.last_loss,_,_ = self.loss_evaluator.evaluate(p1, n2, quiet=True)
        self.last_penalty = self._calculate_thickness_penalty(device)
        
        return self.last_loss + self.last_penalty
    
def setup_and_visualize(ax, title, device_params, light_params, evaluator_obj):
    """
    一个辅助函数，用于根据给定的参数设置场景并进行可视化。
    """
    # 1. 根据参数构建器件和光源
    device = LC_device(**device_params)
    sources = [Point_light_source(**cfg) for cfg in calculate_light_sources_from_params(*light_params)]

    # 2. 追迹光线
    all_points, all_normals = [], []
    for source in sources:
        points, normals = source.trace_ray()
        all_points.append(points)
        all_normals.append(normals)
    
    p0, p1, p2, n3 = device.trace_ray(np.hstack(all_points), np.hstack(all_normals))
    rays = (p0, p1, p2, n3)
    
    # 3. 计算命中点
    _, _, hit_points = evaluator_obj.evaluate(p2, n3, quiet=True)

    # 4. 调用主可视化函数
    visualize_scene(
        ax,
        title=title,
        source_list=sources,
        device=device,
        evaluator=evaluator_obj,
        rays=rays,
        hit_points=hit_points
    )
if __name__ == '__main__':
    
    # --- 1. 定义优化问题的初始配置和边界 ---
    print("Step 1: 正在设置优化问题的初始参数和边界...")
    
    # 定义样条控制点的数量和器件的物理边界
    NUM_UP_CONTROL_POINTS = 12
    NUM_DOWN_CONTROL_POINTS = 12
    DEVICE_X_BOUNDS = [0, 15.2]

    # 上表面的初始参数 (基于 y = 39.54 - x)
    control_x_up = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
    base_up_y = 39.54 - control_x_up
    initial_up_offsets = np.zeros(NUM_UP_CONTROL_POINTS)

    # 下表面的初始参数 (基于 y = 0)
    initial_down_offsets = np.zeros(NUM_DOWN_CONTROL_POINTS)

    # 光源的初始参数
    initial_light_params = [-0.1, -0.6, 0.0, 0.6]
    
    # 组合成完整的初始参数向量
    initial_params_vec = np.concatenate([initial_up_offsets, initial_down_offsets, initial_light_params])

    # 定义优化参数的边界 (现在是为偏移量定义边界)
    up_offset_bounds = [(0, 15)] * NUM_UP_CONTROL_POINTS
    down_offset_bounds = [(0,1)] * NUM_DOWN_CONTROL_POINTS
    light_bounds = [(-0.2, 0), (-1, -0.33), (-0.33, 0.33), (0.33, 1)]
    bounds = up_offset_bounds + down_offset_bounds + light_bounds

    # --- 2. 实例化目标函数和评估器 ---
    evaluator_config = {
        'line_normal': [1, 0], 'line_center': [0, 29], 'line_length': 20.0,
        'weights': [0.7, 0.1, 0.2]
    }
    evaluator = PlanarLossEvaluator(**evaluator_config)
    
    constraints_cfg = {'x_range': DEVICE_X_BOUNDS, 'penalty_weight': 1000.0}
    
    # 这个config用于向Objective类传递不参与优化的静态信息
    device_static_config = {
        'num_up_control_points': NUM_UP_CONTROL_POINTS,
        'num_down_control_points': NUM_DOWN_CONTROL_POINTS,
        'bound': DEVICE_X_BOUNDS,
        'base_up_y_line': base_up_y, # 将基准线传入
        'base_down_y_line': np.zeros(NUM_DOWN_CONTROL_POINTS), # 将基准线传入
        'OptEl_to_world_translation_matrix': np.array([[0], [0]]), # 2D平面中无z轴偏移
        'OptEl_to_world_rotation_matrix': np.eye(2) # 2D平面中无旋转
    }

    # 实例化目标函数 (注意: 假设您的Objective类已更新以处理偏移量)
    objective_func = Objective(LC_device, Point_light_source, evaluator, device_static_config,constraints_cfg)

    # --- 3. 优化前可视化 ---
    print("Step 2: 正在生成优化前的场景可视化...")
    fig_before, ax_before = plt.subplots(figsize=(12, 12))
    
    # 准备可视化所需的参数字典
    initial_device_params = {
        'up_y_coords': base_up_y - initial_up_offsets,
        'down_y_coords': np.zeros(NUM_DOWN_CONTROL_POINTS) + initial_down_offsets,
        'bound': DEVICE_X_BOUNDS,
        'num_control_points_up': NUM_UP_CONTROL_POINTS,
        'num_control_points_down': NUM_DOWN_CONTROL_POINTS
    }
    setup_and_visualize(ax_before, "优化前初始场景", initial_device_params, initial_light_params, evaluator)
    plt.show()

    # --- 4. 运行差分进化优化 ---
    print("\nStep 3: 开始运行差分进化优化...")
    max_generations = 100
    
    pbar = tqdm(total=max_generations, desc="Optimizing")
    def callback(xk,convergence):
        # xk 是当前最佳解的参数，convergence是收敛情况
        pbar.update(1)
        pbar.set_postfix({'convergence': f'{convergence}'})

    result = differential_evolution(
        func=objective_func,
        bounds=bounds,
        maxiter=max_generations,
        popsize=15,
        mutation=(0.5, 1.0), # 使用元组以启用抖动(dithering)
        recombination=0.7,
        strategy='best1bin',
        disp=False,
        callback=callback,
        workers=-1  # 并行计算
    )
    
    # --- 5. 优化后结果打印与可视化 ---
    final_up_offsets = result.x[:NUM_UP_CONTROL_POINTS]
    final_down_offsets = result.x[NUM_UP_CONTROL_POINTS : NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS]
    final_light_params = result.x[NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS:]

    print("\n==============================================")
    print("          优化完成 (Optimization Complete)")
    print("==============================================")
    print(f"\n[+] 最佳损失值 (Lowest Loss): {result.fun:.6f}")
    print("\n[+] 最佳参数 (Optimized Parameters):")
    print(f"  - 上表面偏移量: {np.round(final_up_offsets, 4)}")
    print(f"  - 下表面偏移量: {np.round(final_down_offsets, 4)}")
    print(f"  - 光源参数: {np.round(final_light_params, 4)}")
    print("\n" + "="*46)

    print("\nStep 4: 正在生成优化后的场景可视化...")
    fig_after, ax_after = plt.subplots(figsize=(12, 12))
    
    final_device_params = {
        'up_y_coords': base_up_y - final_up_offsets,
        'down_y_coords': np.zeros(NUM_DOWN_CONTROL_POINTS) + final_down_offsets,
        'bound': DEVICE_X_BOUNDS,
        'num_control_points_up': NUM_UP_CONTROL_POINTS,
        'num_control_points_down': NUM_DOWN_CONTROL_POINTS
    }
    plot_title = (
        f"优化后光学场景\n"
        f"最终损失值: {result.fun:.4f}"
    )
    setup_and_visualize(ax_after, plot_title, final_device_params, final_light_params, evaluator)
    plt.show()