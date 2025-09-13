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
class Objective:
    """
    一个封装器类，连接Scipy优化器和您的评估器。
    新版本整合了罚函数来处理约束。
    """
    def __init__(self, lc_class, initial_rays, loss_evaluator, initial_params_config, constraints_config):
        self.lc_class = lc_class
        self.initial_rays = initial_rays
        self.loss_evaluator = loss_evaluator
        self.num_up_params = len(initial_params_config['up_surface_params'])
        self.initial_params = initial_params_config
        
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

    def __call__(self, params):
        """
        这个方法会被scipy.optimize.minimize调用。
        它接收待优化参数，返回一个包含罚分的总损失值。
        """
        # 1. 从优化器传入的一维数组中解析出曲面参数
        self.initial_params['up_surface_params'] = params[:self.num_up_params]
        self.initial_params['down_surface_params'] = params[self.num_up_params:]
        # 2. 创建光学元件实例 (*** 只在这里创建一次 ***)
        device = self.lc_class(**self.initial_params)
        
        # 3. 执行光线追迹，计算主损失
        points, normals = self.initial_rays
        pl2, n3, pl1, n2, final_mask = device.trace_ray(points, normals)
        primary_loss, _, _ = self.loss_evaluator.evaluate(pl2, n3, quiet=True)

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
        
        # 5. 计算最终总损失
        total_loss = primary_loss + total_penalty
        
        # 6. 存储损失值，以便回调函数可以更新tqdm
        self.last_loss = primary_loss
        self.last_penalty = total_penalty
        
        return total_loss

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

def differential_evolution_Opti():
    """
    使用差分进化 (Differential Evolution) 全局优化算法的运行脚本。
    """
    # --- a. 定义初始参数和配置 (与 nelder_mead_Opti 相同) ---
    surface_params=np.load('PMMA_optimize/output/optimized_params_DE_0913_BEST.npy')
    surface_params[0]=39.54
    initial_params_config = {
        'up_surface_params': surface_params[:12],
        'down_surface_params': surface_params[12:],
        'bound':[15.1,1],
        'OptEl_to_world_translation_matrix': np.array([0, -0.5, 0]).reshape(-1, 1),
    }
    
    opt_light_sources_config = [
        {'position': np.array([7.08, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08-0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
        {'position': np.array([7.08+0.51, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30}
    ]

    vis_light_sources_config = [
        {'position': np.array([7.08, 0, -1.1]).reshape(-1, 1), 'num_rays': 500, 'emission_angle_deg': 30},
    ]
    
    evaluator_config = {
        'plane_normal': np.array([1, 0, 0]),
        'plane_center': np.array([0, 0, 29]),
        'plane_width': 1,
        'plane_height': 20,
        'weights': (0.6, 0.1, 0.3),
        'grid_size': 30
    }
    constraints_config = {
        'x_range': [0, 11.2], 
        'y_range': [0, 1],
        'sample_resolution': 25,
        'penalty_weight': 1000.0
    }

    # --- b. (新增) 为差分进化定义每个参数的边界 (Bounds) ---
    print("--- Defining Bounds for Differential Evolution ---")
    bounds = []
    num_up = len(initial_params_config['up_surface_params'])
    num_down = len(initial_params_config['down_surface_params'])
    
    # 上表面参数边界 (示例)
    up_init = initial_params_config['up_surface_params']
    bounds.append((up_init[0]-0.0001, up_init[0] )) # Z-offset (c0) 允许在初始值附近±5变化
    bounds.append((up_init[1] - 0.0002, up_init[1] + 0.0001)) # x-tilt (c1) 允许在初始值附近±2变化
    bounds.append((up_init[2] - 0.0001, up_init[2] + 0.0001)) # y-tilt (c2) 允许在初始值附近±2变化
    bounds.extend([(-0.00005, 0.00005)] * (num_up - 3))    # 其他高阶项允许在(-0.5, 0.5)之间变化
    
    # 下表面参数边界 (示例)
    down_init = initial_params_config['down_surface_params']
    bounds.append((down_init[0], down_init[0]+0.1)) # Z-offset (c0) 允许在初始值附近±5变化
    bounds.append((down_init[1] - 0.01, down_init[1] + 0.01)) # x-tilt (c1) 允许在初始值附近±2变化
    bounds.append((down_init[2] - 0.01, down_init[2] + 0.01)) # y-tilt (c2) 允许在初
    bounds.extend([(-0.00001, 0.00001)] * (num_down - 3))    # 其他高阶项允许在(-0.5, 0.5)之间变化
    print(f"Total parameters to optimize: {len(bounds)}")

    # --- c. (修改) 优化器配置，切换为差分进化 ---
    optimizer_config = {
        'method': 'differential_evolution', # 明确方法
        'maxiter': 20000,          # 迭代的“代数”
        'popsize': 60,           # 每一代的“种群”大小 (popsize * len(params) 是每代的函数评估次数)
        'tol': 0.000001,             # 收敛容忍度
        'bounds': bounds,        # 传入边界！
        'workers': -1            # 使用所有CPU核心并行计算以加速
    }
    
    # --- d. 优化前的可视化 (与 nelder_mead_Opti 相同) ---
    print("\n--- Visualizing Before Optimization ---")
    initial_lc = LC_device(**initial_params_config)
    vis_sources = [Point_light_source(**cfg) for cfg in vis_light_sources_config]
    evaluator = PlanarLossEvaluator(**evaluator_config)
    visualize_ray_tracing("Initial Parameters", initial_lc, vis_sources, evaluator, rays_to_plot_per_source=20)
    
    # --- e. 运行优化 (调用新的全局优化函数) ---
    result = de_run_opti(
        initial_params_config,
        opt_light_sources_config,
        evaluator_config,
        optimizer_config,
        constraints_config
    )

    # --- f. 打印和可视化最终结果 (与 nelder_mead_Opti 相同) ---
    print("\n" + "="*50 + "\nOptimization Finished!" + "\n" + "="*50)
    print(f"Success: {result.success}\nMessage: {result.message}\nFinal Loss: {result.fun:.6f}")
    
    best_params = result.x
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
    np.save('PMMA_optimize/output/optimized_params_DE.npy', best_params)
    visualize_ray_tracing("Optimized Parameters (DE)", final_lc, vis_sources, evaluator, rays_to_plot_per_source=20)
# ============================================================================
# 步骤 5: 导出stl
# ============================================================================
def generate_lc_device_stl(lc_device: LC_device, output_filename: str, density: int = 100):
    """
    根据LC_device实例生成一个闭合的实体STL模型。

    :param lc_device: LC_device类的一个实例。
    :param output_filename: 输出的STL文件名。
    :param density: XY平面的网格密度，数值越高模型越精细。
    """
    print(f"正在生成STL模型，密度为 {density}x{density}...")

    # --- 1. 生成顶点网格 ---
    x = np.linspace(-lc_device.bound_x / 2., lc_device.bound_x / 2., density)
    y = np.linspace(-lc_device.bound_y / 2., lc_device.bound_y / 2., density)
    xx, yy = np.meshgrid(x, y)

    # --- 2. 计算上下表面顶点 ---
    # 向量化计算所有Z坐标
    zz_up = lc_device.up_surface_fun(xx, yy)
    zz_down = lc_device.down_surface_fun(xx, yy)
    
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
    rotation_matrix = lc_device.OptEl_to_world_rotation_matrix
    translation_vector = lc_device.OptEl_to_world_translation_matrix
    
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


# ==============================================================================
#                                 使用示例
# ============================================================================== 
if __name__ == '__main__':

    # differential_evolution_Opti() # 使用差分进化优化
    # nelder_mead_Opti()            # 使用Nelder-Mead优化
    lcup_surface_params=np.load('PMMA_optimize/output/optimized_params_DE_0913_BEST.npy')[:12]
    lcup_surface_params[0]=39.54
    lcdown_surface_params=np.load('PMMA_optimize/output/optimized_params_DE_0913_BEST.npy')[12:]
    device=LC_device(up_surface_params=lcup_surface_params,
                     down_surface_params=lcdown_surface_params,
                     bound=[14,1],
                     )
    device.generate_lc_device_stl(
        output_filename='PMMA_optimize/output/lc_device_model.stl', 
        density=100  # 提高密度以获得更平滑的模型
    )