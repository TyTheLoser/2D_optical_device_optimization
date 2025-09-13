import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import minimize,differential_evolution
from stl import mesh

from utils.utils_ray_trace import LC_device,Point_light_source
from utils.utils_lossEvaluator import calculate_ray_plane_intersections, PlanarLossEvaluator
def visualize_ray_tracing(title, lc_device, light_sources, evaluator, rays_to_plot_per_source=50):
    """
    一个独立的可视化函数，用于绘制完整的光线追迹场景。

    :param title: str, 图像的标题。
    :param lc_device: LC_device, 需要可视化的光学元件实例。
    :param light_sources: list, 包含一个或多个 Point_light_source 实例的列表。
    :param evaluator: PlanarLossEvaluator, 评估器实例，用于绘制目标平面。
    :param rays_to_plot_per_source: int, 每个光源绘制的光线数量，避免图像过于杂乱。
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

    # a. 绘制光学元件和评估平面
    lc_device.plot_OptEl(ax)
    evaluator.plot_OptEL(ax, plane_color='green')

    # b. 循环处理列表中的每一个光源
    for i, source in enumerate(light_sources):
        source.plot_OptEl(ax) # 绘制光源位置
        
        # 为每个光源独立进行光线追迹
        points, normals = source.trace_ray()
        pl2, n3, pl1, n2, final_mask = lc_device.trace_ray(points, normals)
        
        # 使用 final_mask 筛选出成功跑通全程的初始光线
        initial_points_successful = points[:, final_mask]

        if initial_points_successful.shape[1] > 0:
            # 延长出射光线以便观察
            pl3 = pl2 + n3 * 10
            ray_paths = np.array([initial_points_successful, pl1, pl2, pl3])

            # 为了避免图像过于杂乱，只绘制一部分光线
            num_rays = min(rays_to_plot_per_source, ray_paths.shape[2])
            for i in range(num_rays):
                x, y, z = ray_paths[:, 0, i], ray_paths[:, 1, i], ray_paths[:, 2, i]
                ax.plot(x, y, z, 'b', linewidth=0.5)
                # ax.scatter(x, y, z, c='r', marker='o', s=7) # 交点可以按需显示

    # 设置一个好的观察视角
    ax.view_init(elev=10, azim=-70)
    plt.show()
def calculate_light_sources_from_params(a, l1, l2, l3, num_rays=500, emission_angle_deg=30):
    """
    根据斜率a和三个长度比例l1,l2,l3，计算三个光源的位置和姿态。

    :param a: 直线的斜率 (z+3.25 = a*(x-7.23))
    :param l1, l2, l3: 三个光源在线段上的相对位置比例 [-1, 1]
    :return: light_sources_config 列表
    """
    # --- a. 计算直线与矩形的交点 ---
    rect_bounds = {'x_min': 2.8, 'x_max': 11.66, 'z_min': -5.5, 'z_max': -1.0}
    
    intersections = []
    
    # 直线方程: z = a*x - 7.23*a - 3.25
    # 检查与 x 边界的交点
    x_at_xmin = rect_bounds['x_min']
    z_at_xmin = a * x_at_xmin - 7.23 * a - 3.25
    if rect_bounds['z_min'] <= z_at_xmin <= rect_bounds['z_max']:
        intersections.append(np.array([x_at_xmin, z_at_xmin]))
        
    x_at_xmax = rect_bounds['x_max']
    z_at_xmax = a * x_at_xmax - 7.23 * a - 3.25
    if rect_bounds['z_min'] <= z_at_xmax <= rect_bounds['z_max']:
        intersections.append(np.array([x_at_xmax, z_at_xmax]))

    # 检查与 z 边界的交点
    if a != 0: # 避免除以零
        z_at_zmin = rect_bounds['z_min']
        x_at_zmin = (z_at_zmin + 7.23 * a + 3.25) / a
        if rect_bounds['x_min'] <= x_at_zmin <= rect_bounds['x_max']:
            intersections.append(np.array([x_at_zmin, z_at_zmin]))

        z_at_zmax = rect_bounds['z_max']
        x_at_zmax = (z_at_zmax + 7.23 * a + 3.25) / a
        if rect_bounds['x_min'] <= x_at_zmax <= rect_bounds['x_max']:
            intersections.append(np.array([x_at_zmax, z_at_zmax]))

    # 应该恰好找到两个交点形成线段
    if len(intersections) != 2:
        # 如果找不到两个交点（例如直线完全在矩形外），返回一个空配置或错误
        # 在优化中，这种情况可能会得到一个极差的分数，从而被算法淘汰
        return []
        
    p_start, p_end = intersections
    
    # --- b. 计算线段中心和半长向量 ---
    segment_center = (p_start + p_end) / 2.0
    segment_half_vector = (p_end - p_start) / 2.0

    # --- c. 计算光源位置 ---
    ratios = [l1, l2, l3]
    positions_xz = [segment_center + r * segment_half_vector for r in ratios]
    
    # 转换为三维列向量 (x, 0, z)
    positions_3d = [np.array([pos[0], 0, pos[1]]).reshape(-1, 1) for pos in positions_xz]

    # --- d. 计算光源姿态 (旋转矩阵) ---
    # 方向向量是线段的方向，指向z轴正半轴，所以我们确保dz为正
    direction_vec_xz = p_end - p_start
    if np.linalg.norm(direction_vec_xz) < 1e-9:
        # 如果线段长度几乎为零，则使用默认方向（沿X轴）
        direction_vec_xz = np.array([1.0, 0.0])
    else:
        if direction_vec_xz[1] < 0: # z分量为负
            direction_vec_xz = -direction_vec_xz
        # 将方向向量单位化
        direction_vec_xz = direction_vec_xz / np.linalg.norm(direction_vec_xz)

    # 从单位方向向量中获取 dx 和 dz
    dx, dz = direction_vec_xz[0], direction_vec_xz[1]

    # 构建新的坐标系
    # 新的 X' 轴是线段的方向
    new_x_axis = np.array([dx, 0, dz])
    # 新的 Y' 轴保持不变 (绕Y轴旋转)
    new_y_axis = np.array([0, 1, 0])
    # 新的 Z' 轴垂直于线段方向且“朝上”（在XZ平面内旋转90度）
    new_z_axis = np.array([-dz, 0, dx])

    # 将新的基准坐标轴作为列向量组合成旋转矩阵
    # np.stack([...], axis=1) 可以方便地实现这一点
    rotation_matrix = np.stack([new_x_axis, new_y_axis, new_z_axis], axis=1)

    # 现在 rotation_matrix 会将光源的默认 Z 轴 (0,0,1) 旋转到 new_z_axis 的方向，
    # 同时将默认的 X 轴 (1,0,0) 旋转到 new_x_axis 的方向。
    
    # --- e. 构建光源配置列表 ---
    light_sources_config = []
    for pos in positions_3d:
        light_sources_config.append({
            'OptEl_to_world_translation_matrix': pos,
            'num_rays': num_rays,
            'emission_angle_deg': emission_angle_deg,
            "OptEl_to_world_rotation_matrix": rotation_matrix
        })
        
    return light_sources_config

class Objective:
    """
    一个封装器类，连接Scipy优化器和您的评估器。
    新版本整合了光源参数优化和约束处理。
    """
    def __init__(self, lc_class, light_source_class, loss_evaluator, 
                 initial_params_config, initial_light_control_params, constraints_config):
        
        # 存储构造器类
        self.lc_class = lc_class
        self.light_source_class = light_source_class
        self.loss_evaluator = loss_evaluator
        
        # 记录各部分参数的数量，用于后续解包
        self.num_up_params = len(initial_params_config['up_surface_params'])
        self.num_down_params = len(initial_params_config['down_surface_params'])
        self.num_light_params = len(initial_light_control_params)
        
        # 存储设备不变的参数 (例如 bound, translation)
        self.device_static_config = {
            'bound': initial_params_config['bound'],
            'OptEl_to_world_translation_matrix': initial_params_config['OptEl_to_world_translation_matrix']
        }
        
        # 存储约束检查所需的网格
        cfg = constraints_config
        sample_res = cfg.get('sample_resolution', 20)
        x_samples = np.linspace(cfg['x_range'][0], cfg['x_range'][1], sample_res)
        y_samples = np.linspace(cfg['y_range'][0], cfg['y_range'][1], sample_res)
        self.xx, self.yy = np.meshgrid(x_samples, y_samples)
        
        # 存储罚函数权重和上一次的计算结果
        self.penalty_weight = cfg.get('penalty_weight', 1000.0)
        self.last_loss = float('inf')
        self.last_penalty = float('inf')

    def _calculate_thickness_penalty(self, device):
        """计算厚度约束的罚函数"""
         # 4. (新增) 使用同一个 device 实例计算所有约束的惩罚
        total_penalty = 0.0
        
        # 计算表面Z值 (只计算一次)
        z_up = device.up_surface_fun(self.xx, self.yy)
        z_down = device.down_surface_fun(self.xx, self.yy)
        
        # 约束1: 上表面 > 下表面
        c1 = np.min(z_up - z_down)
        # 如果 c1 < 0, 说明约束被违反, 施加惩罚
        total_penalty += self.penalty_weight * np.maximum(0, -c1)**2

        # 约束2: 上表面 < 39.54 - x
        boundary_z_up = 39.54 - self.xx
        c2 = np.min(boundary_z_up - z_up)
        total_penalty += self.penalty_weight * np.maximum(0, -c2)**2
        
        # 约束3: 下表面 > 0
        c3 = np.min(z_down - 0)
        total_penalty += self.penalty_weight * np.maximum(0, -c3)**2
        
        
        return total_penalty

    def __call__(self, params):
        """
        这个方法会被scipy.optimize调用。
        它接收一个包含所有待优化参数的向量，返回一个总损失值。
        """
        # 1. 从优化器传入的一维数组中解包所有参数
        up_params = params[:self.num_up_params]
        down_params = params[self.num_up_params : self.num_up_params + self.num_down_params]
        light_control_params = params[self.num_up_params + self.num_down_params:]

        # 2. 动态生成当前迭代的光源配置
        # 使用 * 将 light_control_params 列表解包为独立的参数 (a, l1, l2, l3)
        current_light_sources_config = calculate_light_sources_from_params(*light_control_params)

        # 如果光源参数组合无效（例如直线不与矩形相交），立即返回一个巨大的惩罚值
        if not current_light_sources_config:
            self.last_loss = float('inf')
            self.last_penalty = float('inf')
            # 优化器会寻找最小值，所以返回一个大数来“惩罚”这个无效的参数组合
            return 1e9

        # 3. 创建当前迭代的光学系统实例
        # a. 创建光学元件
        device_current_config = {
            'up_surface_params': up_params,
            'down_surface_params': down_params,
            **self.device_static_config
        }
        device = self.lc_class(**device_current_config)

        # b. 创建光源
        light_sources = [self.light_source_class(**config) for config in current_light_sources_config]

        all_initial_points, all_initial_normals = [], []
        for light_source in light_sources:
            points, normals = light_source.trace_ray()
            all_initial_points.append(points); all_initial_normals.append(normals)
        initial_points,initial_normals = np.hstack(all_initial_points), np.hstack(all_initial_normals)
        # 3. 执行光线追迹，计算主损失
        pl2, n3, pl1, n2, final_mask = device.trace_ray(initial_points, initial_normals)
        points_evaluate, normals_evaluate = np.hstack([pl1, pl2]), np.hstack([n2, n3])
        loss, _, _ = self.loss_evaluator.evaluate(points_evaluate, normals_evaluate, quiet=True)
        
        # 5. 计算损失值 (Loss) 和罚分 (Penalty)
        penalty = self._calculate_thickness_penalty(device)
        
        # 存储本次结果，用于调试或打印回调
        self.last_loss = loss
        self.last_penalty = penalty
        
        # 6. 返回加权总分
        total_score = loss + penalty
        
        # (可选) 打印进度
        # print(f"Loss: {loss:.4f}, Penalty: {penalty:.4f}, Total Score: {total_score:.4f}")
        
        return total_score


# ============================================================================
# 步骤 2: (修改) 优化控制器，移除 constraint 参数
# ============================================================================
def nelder_mead_run_opti(
    initial_params_config,
    light_sources_config_list,
    evaluator_config,
    optimizer_config,
    constraints_config, # 约束配置现在传入 Objective 类
):
    """
    执行完整的优化流程。
    """
    # --- 1. 初始化 (大部分不变) ---
    print("--- Initializing Optimization ---")
    # ... (合并光线等代码不变) ...
    all_initial_points, all_initial_normals = [], []
    for ls_config in light_sources_config_list:
        source = Point_light_source(**ls_config)
        points, normals = source.trace_ray()
        all_initial_points.append(points); all_initial_normals.append(normals)
    initial_rays = (np.hstack(all_initial_points), np.hstack(all_initial_normals))
    
    loss_evaluator = PlanarLossEvaluator(**evaluator_config)
    
    # 将约束配置传入 Objective 类
    objective_func = Objective(LC_device, initial_rays, loss_evaluator, initial_params_config, constraints_config)
    
    
    # --- 2. (修改) 设置 TQDM 回调，现在可以显示罚分 ---
    pbar = tqdm(total=optimizer_config.get('maxiter', 100), desc="Optimizing")
    
    def callback(xk):
        pbar.set_postfix({
            'loss': f'{objective_func.last_loss:.4f}',
            'penalty': f'{objective_func.last_penalty:.4f}'
        })
        pbar.update(1)
        # 提前停止逻辑 (如果需要)
        # if objective_func.last_loss < 1.0 and objective_func.last_penalty == 0:
        #     raise StopIteration("Target loss reached with zero penalty.")

    # --- 3. (修改) 执行优化，不再需要 constraints 参数 ---
    print("\n--- Starting Optimization Loop ---")
    result = opt.minimize(
        fun=objective_func,
        x0=optimizer_config.get('x0'),
        method=optimizer_config.get('method', 'Nelder-Mead'), # 现在可以使用任何算法
        tol=optimizer_config.get('tol', 1e-6),
        options={'maxiter': optimizer_config.get('maxiter', 10000), 'disp': False},
        callback=callback
    )    
    pbar.close()
    
    # --- 4. 返回结果 ---
    return result
def de_run_opti(
    initial_params_config,
    light_sources_config_list,
    evaluator_config,
    optimizer_config,
    constraints_config,
):
    """
    使用差分进化 (Differential Evolution) 执行全局优化流程。
    """
    # --- 1. 初始化 (与 run_optimization 类似) ---
    print("--- Initializing Global Optimization (Differential Evolution) ---")
    all_initial_points, all_initial_normals = [], []
    for ls_config in light_sources_config_list:
        source = Point_light_source(**ls_config)
        points, normals = source.trace_ray()
        all_initial_points.append(points); all_initial_normals.append(normals)
    initial_rays = (np.hstack(all_initial_points), np.hstack(all_initial_normals))
    
    loss_evaluator = PlanarLossEvaluator(**evaluator_config)
    objective_func = Objective(LC_device, initial_rays, loss_evaluator, initial_params_config, constraints_config)
    
    # --- 2. 检查和获取差分进化所需的 bounds 参数 ---
    bounds = optimizer_config.get('bounds')
    if bounds is None:
        raise ValueError("Differential evolution requires the 'bounds' parameter in optimizer_config.")
        
    # --- 3. 设置 TQDM 和差分进化专用的回调函数 ---
    # 对于DE，maxiter指的是“代数”，总迭代次数会更多
    pbar = tqdm(total=optimizer_config.get('maxiter', 100), desc="Diff. Evolution")
    
    def de_callback(xk, convergence):
        # xk 是当前最优参数，我们可以用它来获取并显示最新的损失值
        # 为避免重复计算，可以从 objective_func 实例中获取（但这要求 objective_func 在回调中被调用）
        # 一个简单的方法是直接重新计算一次损失用于显示
        loss = objective_func(xk) 
        pbar.set_postfix({
            'loss': f'{objective_func.last_loss:.4f}',
            'penalty': f'{objective_func.last_penalty:.4f}',
            'convergence': f'{convergence:.4f}'
        })
        pbar.update(1)

    # --- 4. 执行差分进化优化 ---
    print("\n--- Starting Differential Evolution Global Optimization Loop ---")
    result = differential_evolution(
        func=objective_func,
        bounds=bounds,
        strategy=optimizer_config.get('strategy', 'best1bin'),
        maxiter=optimizer_config.get('maxiter', 100),
        popsize=optimizer_config.get('popsize', 15),
        tol=optimizer_config.get('tol', 0.01),
        recombination=optimizer_config.get('recombination', 0.7),
        disp=False,
        callback=de_callback,
        workers=optimizer_config.get('workers', -1) # 使用所有可用的CPU核心并行计算
    )
    pbar.close()
    
    return result
# ============================================================================
# 步骤 4: 配置并运行主程序
# ============================================================================
def nelder_mead_Opti():
    """
    使用差分进化 (Nelder Mead) 全局优化算法的运行脚本。
    """
    # --- a. 定义初始参数和配置 ---
    initial_params_config = {
        'up_surface_params': [39.54, -1.4,-0.1,1e-6, 1e-6, 1e-6,1e-6,1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        'down_surface_params': [1e-6, 1e-6,1e-6,1e-6, 1e-6, 1e-6,1e-6,1e-6, 1e-6, 1e-6, 1e-6, 1e-6],
        'bound':[15.1,1],
        'OptEl_to_world_translation_matrix': np.array([0, -0.5, 0]).reshape(-1, 1),
    }
    
    # 为优化过程定义光源 (光线数较少，加快计算速度)
    opt_light_sources_config = [
        {'position': np.array([7.08, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08-0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08+0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30}
    ]

    # 为可视化过程定义光源 (光线数较多，图像更清晰)
    vis_light_sources_config = [
        {'position': np.array([7.08, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08-0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08+0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30}
    ]
    
    evaluator_config = {
        'plane_normal': np.array([1, 0, 0]),
        'plane_center': np.array([0, 0, 29]),
        'plane_width': 1,
        'plane_height': 20,
        'weights': (0.6, 0.1, 0.3), # (平行度, 均匀度, 覆盖度)
        'grid_size': 30
    }
    # (新增) 约束检查的范围和精度
    constraints_config = {
        'x_range': [0, 11.2], 
        'y_range': [0, 1],
        'sample_resolution': 25,
        'penalty_weight': 1000.0 # 罚函数权重，可以调整
    }
    x0 = np.concatenate([
        initial_params_config['up_surface_params'], 
        initial_params_config['down_surface_params']
    ])

    # (修改) 优化器配置，现在可以使用 Nelder-Mead 或其他任何算法
    optimizer_config = {
        'method': 'Nelder-Mead', # 优化方法
        'maxiter': 10000 , # 最大迭代次数
        'tol': 1e-6, # 收敛容忍度
        'x0': x0 # 优化对象的初始参数
    }
    
    

    # --- b. 优化前的可视化 ---
    print("--- Visualizing Before Optimization ---")
    initial_lc = LC_device(**initial_params_config)
    vis_sources = [Point_light_source(**cfg) for cfg in vis_light_sources_config]
    evaluator = PlanarLossEvaluator(**evaluator_config)
    visualize_ray_tracing("Initial Parameters", initial_lc, vis_sources, evaluator, rays_to_plot_per_source=20)
    
    # --- c. 运行优化 ---
    result = nelder_mead_run_opti(
        initial_params_config,
        opt_light_sources_config,
        evaluator_config,
        optimizer_config,
        constraints_config
    )

    # --- d. 打印和可视化最终结果 ---
    print("\n" + "="*50 + "\nOptimization Finished!" + "\n" + "="*50)
    print(f"Success: {result.success}\nMessage: {result.message}\nFinal Loss: {result.fun:.6f}")
    
    best_params = result.x
    num_up = len(initial_params_config['up_surface_params'])
    best_up_params = best_params[:num_up]
    best_down_params = best_params[num_up:]
    
    print(f"\nBest Up-Surface Parameters:\n{np.round(best_up_params, 4)}")
    print(f"Best Down-Surface Parameters:\n{np.round(best_down_params, 4)}")
    print("="*50 + "\n")
    
    print("--- Visualizing After Optimization ---")
    final_params_config = initial_params_config.copy()
    final_params_config['up_surface_params'] = best_up_params
    final_params_config['down_surface_params'] = best_down_params
    final_lc = LC_device(**final_params_config)
    # 可视化依然使用 vis_sources
    visualize_ray_tracing("Optimized Parameters", final_lc, vis_sources, evaluator, rays_to_plot_per_source=20)

def differential_evolution_Opti(
    initial_params_path: str=None,
    save_path: str = None,
    bounds: list = None,
    visualize_before_run: bool = True
):
    """
    使用差分进化 (Differential Evolution) 全局优化算法的运行脚本。
    此函数现在支持表面和光源参数的联合优化。

    :param initial_params_path: 包含初始参数的一维Numpy数组 (.npy) 的路径。
                                 格式: [12个上表面参数, 12个下表面参数, 4个光源控制参数]
    :param save_path: 优化后的最佳参数的保存路径 (.npy)。如果为None，则默认覆盖初始文件。
    :param bounds: Scipy优化器所需的边界列表。如果为None，则根据初始参数自动生成。
    :param visualize_before_run: 是否在优化开始前进行可视化。
    """
    # --- 1. 处理输入参数和路径 ---
    if save_path is None:
        save_path = initial_params_path
    
    print(f"--- Starting Differential Evolution Optimization ---")
    print(f"Loading initial parameters from: {initial_params_path}")
    print(f"Optimized parameters will be saved to: {save_path}")

    # --- 2. 定义不变的配置 ---
    # 这些配置定义了本次优化的“问题”本身，保持不变
    evaluator_config = {
        'plane_normal': np.array([1, 0, 0]), 'plane_center': np.array([0, 0, 29]),
        'plane_width': 1, 'plane_height': 20, 'weights': (0.6, 0.1, 0.3), 'grid_size': 30
    }
    constraints_config = {
        'x_range': [0, 11.2], 'y_range': [0, 1],
        'sample_resolution': 25, 'penalty_weight': 1000.0
    }
    optimizer_static_config = {
        'maxiter': 1, 'popsize': 20, 'tol': 0.000001, 'workers': -1
    }

    
    
    
    # --- 3. 加载并解包初始参数 ---
    if initial_params_path is None:
        print("Initial parameters config not provided, using default values.")
        print("auto-generating default initial parameters...")
        initial_params_vector = np.concatenate([
            np.array([39.54, -1.4,-0.1] + [0]*9), # 上表面
            np.array([0]*12),                    # 下表面
            np.array([-0.1, 0.0, -0.5, 0.5])       # 光源控制参数 (a, l1, l2, l3)
        ])
    else:
        initial_params_vector = np.load(initial_params_path)
    # 定义参数结构
    num_up, num_down, num_light = 12, 12, 4
    if len(initial_params_vector) != num_up + num_down + num_light:
        print(f"Initial parameters file has wrong length. Expected {num_up+num_down+num_light}, got {len(initial_params_vector)}")
        print("auto-generating default initial parameters...")
        initial_params_vector = np.concatenate([
            np.array([39.54, -1.4,-0.1] + [0]*9), # 上表面
            np.array([0]*12),                    # 下表面
            np.array([-0.1, 0.0, -0.5, 0.5])       # 光源控制参数 (a, l1, l2, l3)
        ])

    up_init = initial_params_vector[:num_up]
    down_init = initial_params_vector[num_up : num_up + num_down]
    light_init = initial_params_vector[num_up + num_down:]

    initial_params_config = {
        'up_surface_params': up_init,
        'down_surface_params': down_init,
        'bound': [15.1, 1],
        'OptEl_to_world_translation_matrix': np.array([0, -0.5, 0]).reshape(-1, 1),
    }

    # --- 4. 设置优化边界 (Bounds) ---
    if bounds is None:
        print("--- No bounds provided, generating default bounds... ---")
        bounds = []
        # 上表面边界
        bounds.append((up_init[0] - 0.01, up_init[0]+1e-6))
        bounds.append((up_init[1] - 0.5, up_init[1]+1e-6))
        bounds.append((up_init[2] - 0.1, up_init[2]+1e-6))
        bounds.extend([(-0.00005, 0.00005)] * (num_up - 3))
        # 下表面边界
        bounds.append((down_init[0], down_init[0] + 0.1))
        bounds.append((down_init[1] - 0.01, down_init[1] + 0.01))
        bounds.append((down_init[2] - 0.01, down_init[2] + 0.01))
        bounds.extend([(-0.00001, 0.00001)] * (num_down - 3))
        # 光源参数边界
        bounds.append((-1.0, 1))  # a
        bounds.extend([(-1.0, 1.0)] * 3) # l1, l2, l3

    print(f"Total parameters to optimize: {len(bounds)}")

    # --- 5. 实例化 Objective 评估器 ---
    # 确保将所有需要的类和配置传入
    objective_instance = Objective(
        lc_class=LC_device,
        light_source_class=Point_light_source,
        loss_evaluator=PlanarLossEvaluator(**evaluator_config),
        initial_params_config=initial_params_config,
        initial_light_control_params=light_init,
        constraints_config=constraints_config
    )

    # --- 6. (可选) 优化前可视化 ---
    if visualize_before_run:
        print("\n--- Visualizing Before Optimization ---")
        initial_lc = LC_device(**initial_params_config)
        # 动态生成用于可视化的光源
        vis_light_sources_config = calculate_light_sources_from_params(*light_init)
        vis_sources = [Point_light_source(**cfg) for cfg in vis_light_sources_config]
        evaluator = PlanarLossEvaluator(**evaluator_config)
        visualize_ray_tracing("Initial State", initial_lc, vis_sources, evaluator, rays_to_plot_per_source=20)

    # --- 7. 设置 TQDM 和回调函数 ---
    pbar = tqdm(total=optimizer_static_config.get('maxiter', 100), desc="Diff. Evolution")

    def de_callback(xk, convergence):
        """
        在每一代优化结束时被调用，用于更新进度条。
        """
        loss = objective_instance(xk) 
        pbar.set_postfix({
            'loss': f'{objective_instance.last_loss:.4f}',
            'penalty': f'{objective_instance.last_penalty:.4f}',
            'convergence': f'{convergence:.4f}'
        })
        pbar.update(1)
    # --- 7. 运行优化 ---
    print("\n--- Starting Scipy Differential Evolution ---")
    result = differential_evolution(
        func=objective_instance,
        bounds=bounds,
        x0=initial_params_vector,
        callback=de_callback, # 传入回调函数
        **optimizer_static_config
    )
    pbar.close() # 确保在优化结束后关闭进度条

    # --- 8. 打印、保存和可视化最终结果 ---
    print("\n" + "="*50 + "\nOptimization Finished!" + "\n" + "="*50)
    print(f"Success: {result.success}\nMessage: {result.message}\nFinal Score: {result.fun:.6f}")
    
    best_params = result.x
    np.save(save_path, best_params)
    print(f"\nBest parameters saved to: {save_path}")

    best_up_params = best_params[:num_up]
    best_down_params = best_params[num_up : num_up + num_down]
    best_light_params = best_params[num_up + num_down:]
    
    print(f"\nBest Up-Surface Parameters:\n{np.round(best_up_params, 5)}")
    print(f"Best Down-Surface Parameters:\n{np.round(best_down_params, 5)}")
    print(f"Best Light Control Parameters (a, l1, l2, l3):\n{np.round(best_light_params, 5)}")
    print("="*50 + "\n")
    
    print("--- Visualizing After Optimization ---")
    final_params_config = initial_params_config.copy()
    final_params_config['up_surface_params'] = best_up_params
    final_params_config['down_surface_params'] = best_down_params
    final_lc = LC_device(**final_params_config)
    
    # 使用优化后的最佳参数动态生成光源用于最终可视化
    final_light_sources_config = calculate_light_sources_from_params(*best_light_params)
    final_sources = [Point_light_source(**cfg) for cfg in final_light_sources_config]
    final_evaluator = PlanarLossEvaluator(**evaluator_config)
    
    visualize_ray_tracing("Optimized State (DE)", final_lc, final_sources, final_evaluator, rays_to_plot_per_source=20)

    return result

# ==============================================================================
#                                 使用示例
# ============================================================================== 
if __name__ == '__main__':

    # INITIAL_PARAMS_FILE = 'PMMA_optimize/output/initial_params_with_light.npy'
    OPTIMIZED_PARAMS_FILE = 'PMMA_optimize/output/optimized_params_final.npy'


    # 调用重写后的优化函数
    differential_evolution_Opti(
        save_path=OPTIMIZED_PARAMS_FILE,
        visualize_before_run=False # 在正式运行时可以设为 False 以节省时间
    )

    # nelder_mead_Opti()            # 使用Nelder-Mead优化

    #导出stl
    # lcup_surface_params=np.load('PMMA_optimize/output/optimized_params_DE_0913_BEST.npy')[:12]
    # lcup_surface_params[0]=39.54
    # lcdown_surface_params=np.load('PMMA_optimize/output/optimized_params_DE_0913_BEST.npy')[12:]
    # device=LC_device(up_surface_params=lcup_surface_params,
    #                  down_surface_params=lcdown_surface_params,
    #                  bound=[14,1],
    #                  )
    # device.generate_lc_device_stl(
    #     output_filename='PMMA_optimize/output/lc_device_model.stl', 
    #     density=100  # 提高密度以获得更平滑的模型
    # )