import numpy as np
import scipy.optimize as opt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.utils_ray_trace import LC_device,Point_light_source
from utils.utils_lossEvaluator import calculate_ray_plane_intersections, PlanarLossEvaluator
def run_optimization(
    initial_up_params,
    initial_down_params,
    light_source_config,
    evaluator_config,
    optimizer_config
):
    """
    执行光学系统的优化流程。

    :param initial_up_params: list, 上表面的初始参数。
    :param initial_down_params: list, 下表面的初始参数。
    :param light_source_config: dict, 点光源的配置。
    :param evaluator_config: dict, 损失评估器的配置。
    :param optimizer_config: dict, Scipy优化器的配置。
    :return: 优化结果对象。
    """
    
    # --- 初始化 ---
    # 将要优化的参数合并为一个一维数组
    initial_params = np.concatenate([initial_up_params, initial_down_params])
    num_up_params = len(initial_up_params)
    
    # 初始化光源和初始光线 (在优化循环外执行一次即可)
    pls = Point_light_source(**light_source_config)
    initial_points, initial_normals = pls.trace_ray()
    
    # 初始化tqdm进度条
    pbar = tqdm(total=optimizer_config.get('maxiter', 100), desc="Optimizing")
    
    # --- 定义优化所需的目标函数和回调函数 ---
    
    def objective_func(params):
        """
        目标函数，scipy.optimize.minimize 将尝试最小化此函数的返回值。
        """
        # 1. 从一维数组中解析出上下表面的参数
        current_up_params = params[:num_up_params]
        current_down_params = params[num_up_params:]
        
        # 2. 使用当前参数创建光学元件
        lc = LC_device(current_up_params, current_down_params)
        
        # 3. 执行光线追迹
        pl2_wcs, n3_wcs, pl1_wcs, n2_wcs = lc.trace_ray(initial_points, initial_normals)

        # 4. 初始化损失评估器
        evaluator = PlanarLossEvaluator(**evaluator_config)
        
        # 5. 评估三组不同的光线，并计算总损失
        #    - quiet=True 避免在每次迭代时打印 evaluator 的内部信息
        loss1, losses1 = evaluator.evaluate(initial_points, initial_normals, quiet=True)
        loss2, losses2 = evaluator.evaluate(pl1_wcs, n2_wcs, quiet=True)
        loss3, losses3 = evaluator.evaluate(pl2_wcs, n3_wcs, quiet=True)

        # 合并损失，这里简单相加，也可以加权
        total_loss = loss1 + loss2 + loss3
        
        # 6. 更新tqdm进度条的后缀信息
        pbar.set_postfix({
            'Total Loss': f'{total_loss:.4f}',
            'L1(Source)': f'{loss1:.3f}',
            'L2(Mid)': f'{loss2:.3f}',
            'L3(Exit)': f'{loss3:.3f}',
            'P_exit': f'{losses3[0]:.3f}', # 显示最终出射光的平行度损失
            'U_exit': f'{losses3[1]:.3f}', # 显示最终出射光的均匀度损失
            'C_exit': f'{losses3[2]:.3f}'  # 显示最终出射光的覆盖度损失
        })
        
        return total_loss

    def callback_func(xk):
        """
        每次迭代完成后的回调函数，仅用于更新进度条。
        """
        pbar.update(1)

    # --- 执行优化 ---
    result = minimize(
        fun=objective_func,
        x0=initial_params,
        method=optimizer_config.get('method', 'Nelder-Mead'),
        options={'maxiter': optimizer_config.get('maxiter', 100), 'disp': True},
        callback=callback_func
    )
    
    pbar.close()
    
    return result

# ============================================================================
# 步骤 3: 配置并调用优化函数
# ============================================================================
if __name__ == '__main__':
    # --- 1. 定义初始参数和配置 ---
    
    # a. 光学元件的初始曲面参数
    up_params_initial = [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    down_params_initial = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    
    # b. 光源配置
    light_config = {
        'position': np.array([0, 0, -5]).reshape(-1, 1),
        'num_rays': 1000, # 为了快速迭代，优化时可以使用较少光线
        'emission_angle_deg': 5
    }
    
    # c. 损失评估器配置 (目标平面)
    evaluator_config = {
        'plane_normal': np.array([0, 0, 1]),
        'plane_center': np.array([0, 0, 10]),
        'plane_width': 10,
        'plane_height': 10,
        'weights': (0.5, 0.3, 0.2), # (Parallelism, Uniformity, Coverage)
        'grid_size': 30
    }
    
    # d. 优化器配置
    optimizer_config = {
        'method': 'Nelder-Mead', # 一个稳健的无梯度优化算法
        'maxiter': 50 # 最大迭代次数
    }

    # --- 2. 运行优化流程 ---
    optimization_result = run_optimization(
        up_params_initial,
        down_params_initial,
        light_config,
        evaluator_config,
        optimizer_config
    )
    
    # --- 3. 显示优化结果 ---
    print("\n" + "="*50)
    print("Optimization Finished!")
    print(f"Success: {optimization_result.success}")
    print(f"Message: {optimization_result.message}")
    print(f"Final Loss: {optimization_result.fun:.6f}")
    
    best_params = optimization_result.x
    num_up = len(up_params_initial)
    best_up_params = best_params[:num_up]
    best_down_params = best_params[num_up:]
    
    print("\nBest Up-Surface Parameters:")
    print(np.round(best_up_params, 4))
    print("\nBest Down-Surface Parameters:")
    print(np.round(best_down_params, 4))
    print("="*50 + "\n")

    # --- 4. 使用最优参数进行可视化验证 ---
    print("Visualizing result with best parameters...")
    final_lc = LC_device(best_up_params, best_down_params)
    final_pls = Point_light_source(**light_config, num_rays=5000) # 使用更多光线进行可视化
    point, normal = final_pls.trace_ray()
    
    pl2_wcs, n3_wcs, pl1_wcs, n2_wcs = final_lc.trace_ray(point, normal)
    pl3_wcs = pl2_wcs + n3_wcs * 10 # 延长出射光线以便观察
    
    ray_path_points = np.array([point, pl1_wcs, pl2_wcs, pl3_wcs])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制50条光路用于展示
    num_rays_to_plot = min(50, light_config['num_rays'])
    for i in range(num_rays_to_plot):
        x = ray_path_points[:, 0, i]
        y = ray_path_points[:, 1, i]
        z = ray_path_points[:, 2, i]
        ax.plot(x, y, z, 'b', linewidth=0.5)
        ax.scatter(x, y, z, c='r', marker='o', s=7)
        
    final_lc.plot_OptEl(ax)
    final_pls.plot_OptEl(ax)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("Ray Tracing with Optimized Parameters")
    plt.show()
