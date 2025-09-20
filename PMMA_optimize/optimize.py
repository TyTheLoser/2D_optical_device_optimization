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
    def __init__(self, lc_class, light_source_class, loss_evaluator, device_config, light_params, constraints_config):
        self.lc_class, self.light_source_class, self.loss_evaluator = lc_class, light_source_class, loss_evaluator
        self.num_up = len(device_config['up_surface_params']); self.num_down = len(device_config['down_surface_params'])
        self.device_static_cfg = {'bound': device_config['bound'], 'OptEl_to_world_translation_matrix': device_config['OptEl_to_world_translation_matrix']}
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
        up_params, down_params, light_params = params[:self.num_up], params[self.num_up:self.num_up+self.num_down], params[self.num_up+self.num_down:]
        light_configs = calculate_light_sources_from_params(*light_params)
        if not light_configs: return 1e9
        
        device_cfg = {'up_surface_params':up_params, 'down_surface_params':down_params, **self.device_static_cfg}
        device = self.lc_class(**device_cfg)
        
        sources = [self.light_source_class(**cfg) for cfg in light_configs]
        all_p, all_n = zip(*[s.trace_ray() for s in sources])
        p_initial, n_initial = np.hstack(all_p), np.hstack(all_n)
        
        _,_, p1, n2 = device.trace_ray(p_initial, n_initial) # 简化追迹用于评估
        self.last_loss,_,_ = self.loss_evaluator.evaluate(p1, n2, quiet=True)
        self.last_penalty = self._calculate_thickness_penalty(device)
        return self.last_loss + self.last_penalty


if __name__ == '__main__':
    # --- 1. 定义优化问题的各项配置 ---
    device_config = {
        'up_surface_params': [39,-1,0,0,0,0,0,0,0,0,0,0], # y = 1.5
        'down_surface_params': [0, 0,0,0,0,0,0,0,0,0,0,0], # y = -1.5 - 0.01x^2
        'bound': [0,15.2],
        'OptEl_to_world_translation_matrix': np.array([0, 0]).reshape(2, 1)
    }
    initial_light_params = [-0.1, -0.6, 0.0, 0.6] # a, l1, l2, l3
    evaluator = PlanarLossEvaluator(line_normal=[1,0], line_center=[0,29], line_length=20, weights=[0.4,0.3,0.3])
    constraints_cfg = {'x_range': [0, 15.2], 'penalty_weight': 1000.0}
    
    initial_params_vec = np.concatenate([device_config['up_surface_params'], device_config['down_surface_params'], initial_light_params])
    
    bounds = [(39,39.54), (-2,-1), (-0.05,0.05), (-0.05,0.05), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005), (-0.0005,0.0005)] + \
             [(0,2), (0,0.1), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005), (0,0.0005)] + \
             [(-0.2, 0), (-1, -0.33), (-0.33, 0.33), (0.33, 1)]

    # --- 2. 优化前可视化 ---
    print("--- 优化前状态可视化 ---")
    fig_before, ax_before = plt.subplots(figsize=(10, 10))
    initial_device = LC_device(**device_config)
    initial_sources = [Point_light_source(**cfg) for cfg in calculate_light_sources_from_params(*initial_light_params)]
    # --- 2. Trace and Combine Rays ---
    all_initial_points = []
    all_initial_normals = []

    # Iterate through each source in the list
    for source in initial_sources:
        # Trace rays for a single source
        points, normals = source.trace_ray()
        
        # Append the results to the lists
        all_initial_points.append(points)
        all_initial_normals.append(normals)

    all_initial_points = np.hstack(all_initial_points)
    all_initial_normals = np.hstack(all_initial_normals)
    # 追迹光线通过透镜，得到所有路径点和方向向量
    p0, p1, p2, n3 = initial_device.trace_ray(all_initial_points, all_initial_normals)

    # 重新组织光线数据，以匹配 visualize_scene 函数的输入
    rays = (p0, p1, p2, n3)

    # --- 3. 计算命中点 ---
    # 使用评估器计算光线与平面的交点
    _, _, hit_points = evaluator.evaluate(p2, n3, quiet=True)

    # --- 4. 调用 visualize_scene 函数进行可视化 ---
    visualize_scene(
        ax_before,
        title="完整光学场景可视化",
        source_list=initial_sources,
        device=initial_device,
        evaluator=evaluator,
        rays=rays,
        hit_points=hit_points
    )

    plt.show()

    # --- 3. 运行差分进化优化 ---
    objective_func = Objective(LC_device, Point_light_source, evaluator, device_config, initial_light_params, constraints_cfg)
    # --- b. 设置优化器超参数 ---
    max_generations = 10
    
    # --- c. 创建并配置进度条的回调函数 ---
    pbar = tqdm(total=max_generations, desc="Optimizing")
    def callback(xk,convergence):
        # xk 是当前最佳解的参数，convergence是收敛情况
        pbar.update(1)
        pbar.set_postfix({'convergence': f'{convergence}'})

    # --- d. 调用 SciPy 的差分进化函数 ---
    result = differential_evolution(
        func=objective_func,          # 目标函数
        bounds=bounds,                # 参数边界
        maxiter=max_generations,      # 最大迭代次数 (代数)
        popsize=15,                   # 种群大小乘数 (总种群 = popsize * dims)
        mutation=0.8,                 # 变异因子 F
        recombination=0.9,            # 交叉概率 CR
        strategy='best1bin',          # 差分策略 (与您代码中的经典策略一致)
        disp=False,                   # 不在控制台打印收敛信息
        callback=callback,            # 每一次迭代后调用的函数
        workers=-1                    # 使用所有可用的CPU核心进行并行计算
    )

    
    # --- 4. 优化后可视化 ---
    print("\n\n优化完成!")
    print("="*30)
    print(f"最佳损失值 (Best Fitness): {result.fun:.6f}")
    print(f"最佳解 (Best Solution):")
    print(result.x)
    print("\n--- 优化后状态可视化 ---")
    num_up = len(device_config['up_surface_params']); num_down = len(device_config['down_surface_params'])
    final_device_cfg = device_config.copy()
    final_device_cfg['up_surface_params'] = result.x[:num_up]
    final_device_cfg['down_surface_params'] = result.x[num_up:num_up+num_down]
    final_light_params = result.x[num_up+num_down:]
    
    
    final_device = LC_device(**final_device_cfg)
    final_sources = [Point_light_source(**cfg) for cfg in calculate_light_sources_from_params(*final_light_params)]
    # --- 2. Trace and Combine Rays ---
    all_final_points = []
    all_final_normals = []
    for source in final_sources:
        points, normals = source.trace_ray()
        all_final_points.append(points)
        all_final_normals.append(normals)
    
    all_final_points = np.hstack(all_final_points)
    all_final_normals = np.hstack(all_final_normals)
    p0, p1, p2, n3 = final_device.trace_ray(all_final_points, all_final_normals)
    rays = (p0, p1, p2, n3)
    fig_after, ax_after = plt.subplots(figsize=(10, 10))
    visualize_scene(
        ax_after,
        title="优化后的完整光学场景可视化",
        source_list=final_sources,
        device=final_device,
        evaluator=evaluator,
        rays=rays,
        hit_points=hit_points
    )
    plt.show()