import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.collections import LineCollection
import time
import os
from scipy.interpolate import CubicSpline
from utils.utils_ray_trace import LC_device, Point_light_source,calculate_light_sources_from_params
from utils.utils_lossEvaluator import PlanarLossEvaluator, calculate_ray_line_intersections
from utils.utils_free_form_prism_design import visualize_scene
from scipy.optimize import differential_evolution



# =============================================================================
# 3. 动态光源生成与优化目标封装 (2D版本)
# =============================================================================


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
# =============================================================================
#                         --- 主执行流程 ---
# =============================================================================
if __name__ == '__main__':
    
    # =========================================================================
    #  ✅ Step 1: 配置区 (在此处修改所有参数)
    # =========================================================================
    print("Step 1: 正在设置优化问题的配置...")

    # --- a. 文件与几何配置 ---
    NUM_UP_CONTROL_POINTS = 7  # <-- 推荐使用奇数以方便对称
    NUM_DOWN_CONTROL_POINTS = 7 # <-- 推荐使用奇数以方便对称
    DEVICE_X_BOUNDS = [0, 16]
    
    PARAMS_FILE = 'PMMA_optimize/output/bottom_3_lights_116_7_BEST.npz' 
    OUTPUT_PARAMS_FILE = f'PMMA_optimize/output/1011/bottom_3_lights_116_{NUM_UP_CONTROL_POINTS}_0.npz'
    
    ENFORCE_SYMMETRY = True  # <-- 总开关：是否强制曲面左右对称

    # --- b. 光源与评估器配置 ---
    initial_light_params_defaults = [-2.557, 0,0.5] # y坐标, l1
    evaluator_config = { 
        'line_normal': [0, 1], 'line_center': [DEVICE_X_BOUNDS[1]/2, 19], 'line_length': 16, 
        'weights': [0.8, 0.1, 0.1] # 平行度, 均匀性, 覆盖度
    }
    constraints_cfg = {'x_range': DEVICE_X_BOUNDS, 'penalty_weight': 1000.0}
    
    # --- c. 定义 *所有* 参数的完整边界 ---
    up_offset_bounds = [(-3, 5)] * NUM_UP_CONTROL_POINTS #上表面边界
    down_offset_bounds = [(0, 5)] * NUM_DOWN_CONTROL_POINTS #下表面边界
    light_bounds = [(-3.6, -1), (0, 1), (0, 1)]
    full_bounds = up_offset_bounds + down_offset_bounds + light_bounds

    # --- d. 参数冻结配置 (作用于对称/非对称模式) ---
    # True = 参与优化 (Active), False = 冻结 (Frozen)
    up_active_mask = [True] * NUM_UP_CONTROL_POINTS
    down_active_mask = [True] * NUM_DOWN_CONTROL_POINTS
    light_active_mask = [False, False, True] # 示例：冻结所有光源参数

    full_active_mask = np.array(up_active_mask + down_active_mask + light_active_mask)
    
    # --- e. 优化器超参数 ---
    max_generations = 10000
    popsize_multiplier = 20
    
    # =========================================================================
    #  ✅ Step 2: 初始化参数 (加载基准线 或 默认生成)
    # =========================================================================
    print("\nStep 2: 正在初始化优化起点...")
    
    # 定义基准线
    base_down_y = 3.65 #下表面初始形状
    control_x_up = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
    base_up_y = 16 # 上表面初始形状

    expected_len = NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS + len(light_bounds)

    if os.path.exists(PARAMS_FILE):
        try:
            print(f"--> 成功从 '{PARAMS_FILE}' 加载上次的优化结果作为新基准。")
            data = np.load(PARAMS_FILE)
            loaded_up_y = data['up_y_coords']
            loaded_down_y = data['down_y_coords']
            loaded_num_up = int(data['num_up'])
            loaded_num_down = int(data['num_down'])

            # 如果控制点数量发生变化，则通过插值扩展基准线
            if loaded_num_up != NUM_UP_CONTROL_POINTS:
                print(f"--> 上表面维度不匹配，正在从 {loaded_num_up} 点插值为 {NUM_UP_CONTROL_POINTS} 点...")
                old_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], loaded_num_up)
                temp_spline = CubicSpline(old_control_x, loaded_up_y)
                new_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
                base_up_y = temp_spline(new_control_x)
            else:
                base_up_y = loaded_up_y

            if loaded_num_down != NUM_DOWN_CONTROL_POINTS:
                print(f"--> 下表面维度不匹配，正在从 {loaded_num_down} 点插值为 {NUM_DOWN_CONTROL_POINTS} 点...")
                old_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], loaded_num_down)
                temp_spline = CubicSpline(old_control_x, loaded_down_y)
                new_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_DOWN_CONTROL_POINTS)
                base_down_y = temp_spline(new_control_x)
            else:
                base_down_y = loaded_down_y

            initial_light_params = data.get('light_params', initial_light_params_defaults)
        except Exception as e:
            print(f"--> 加载文件 '{PARAMS_FILE}' 出错: {e}。将使用默认起点。")
            initial_light_params = initial_light_params_defaults
    else:
        print(f"--> 未找到 '{PARAMS_FILE}'。将使用默认的硬编码基准线作为起点。")
        initial_light_params = initial_light_params_defaults

    # 本轮优化的初始偏移量都清零
    initial_up_offsets = np.zeros(NUM_UP_CONTROL_POINTS)
    initial_down_offsets = np.zeros(NUM_DOWN_CONTROL_POINTS)
    full_initial_params_vec = np.concatenate([initial_up_offsets, initial_down_offsets, initial_light_params])
    
    # =========================================================================
    #  ✅ Step 3: 准备优化 (处理对称与冻结)
    # =========================================================================
    
    # 实例化核心组件
    evaluator = PlanarLossEvaluator(**evaluator_config)
    device_static_config = {'num_up_control_points': NUM_UP_CONTROL_POINTS, 'num_down_control_points': NUM_DOWN_CONTROL_POINTS, 'bound': DEVICE_X_BOUNDS, 'base_up_y_line': base_up_y, 'base_down_y_line': base_down_y}
    objective_func = Objective(LC_device, Point_light_source, evaluator, device_static_config, constraints_cfg)

    # --- 包装目标函数以处理对称和冻结 ---
    if ENFORCE_SYMMETRY and NUM_UP_CONTROL_POINTS % 2 != 0 and NUM_DOWN_CONTROL_POINTS % 2 != 0:
        print("--> 已启用对称约束：将只优化一半的曲面控制点。")
        num_unique_up = (NUM_UP_CONTROL_POINTS + 1) // 2
        num_unique_down = (NUM_DOWN_CONTROL_POINTS + 1) // 2

        # 提取独立参数的掩码、边界和初始值
        up_active_unique_mask = up_active_mask[:num_unique_up]
        down_active_unique_mask = down_active_mask[:num_unique_down]
        active_unique_mask = np.concatenate([up_active_unique_mask, down_active_unique_mask, light_active_mask])
        
        unique_bounds = up_offset_bounds[:num_unique_up] + down_offset_bounds[:num_unique_down] + light_bounds
        active_bounds = [b for i, b in enumerate(unique_bounds) if active_unique_mask[i]]

        initial_unique_params = np.concatenate([initial_up_offsets[:num_unique_up], initial_down_offsets[:num_unique_down], initial_light_params])
        initial_active_params = initial_unique_params[active_unique_mask]

        def wrapped_objective_func(active_unique_params):
            full_params = full_initial_params_vec.copy()
            # 重新构造完整的、对称的、且包含冻结值的参数向量
            temp_unique_params = initial_unique_params.copy()
            temp_unique_params[active_unique_mask] = active_unique_params
            
            unique_up, unique_down, light_p = np.split(temp_unique_params, [num_unique_up, num_unique_up + num_unique_down])
            
            full_up_offsets = np.concatenate([unique_up, np.flip(unique_up[:-1])])
            full_down_offsets = np.concatenate([unique_down, np.flip(unique_down[:-1])])
            
            full_params = np.concatenate([full_up_offsets, full_down_offsets, light_p])
            return objective_func(full_params)
    else:
        if ENFORCE_SYMMETRY:
            print("--> 警告：控制点数量为偶数，对称约束未启用。")
        print("--> 未启用对称约束：将优化所有活动参数。")
        active_bounds = [b for i, b in enumerate(full_bounds) if full_active_mask[i]]
        initial_active_params = full_initial_params_vec[full_active_mask]
        
        def wrapped_objective_func(active_params):
            full_params = full_initial_params_vec.copy()
            full_params[full_active_mask] = active_params
            return objective_func(full_params)
    
    # =========================================================================
    #  Step 4: 优化前可视化
    # =========================================================================
    print("\nStep 4: 正在生成优化前的场景可视化...")
    fig_before, ax_before = plt.subplots(figsize=(12, 12))
    up_offsets, down_offsets, light_params = np.split(full_initial_params_vec, [NUM_UP_CONTROL_POINTS, NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS])
    initial_device_params = {
        'up_y_coords': base_up_y - up_offsets,
        'down_y_coords': base_down_y + down_offsets,
        'bound': DEVICE_X_BOUNDS,
        'num_control_points_up': NUM_UP_CONTROL_POINTS,
        'num_control_points_down': NUM_DOWN_CONTROL_POINTS
    }
    plot_title = ( 
        f"Before Optimize\n"    
        f"loss: {wrapped_objective_func(initial_active_params):.4f}"    
    )
    setup_and_visualize(ax_before, plot_title, initial_device_params, light_params, evaluator)
    plt.show()

    # =========================================================================
    #  Step 5: 运行差分进化优化
    # =========================================================================
    print(f"\nStep 5: 开始运行差分进化优化... (优化 {len(active_bounds)} 个活动参数)")
    
    pbar = tqdm(total=max_generations, desc="Optimizing")
    def callback(intermediate_result):
        """
        接收一个包含当前优化状态的 OptimizeResult 对象。
        intermediate_result.fun 就是当前找到的最佳损失值。
        """
        pbar.update(1)
        # 直接从 intermediate_result 对象获取损失值并显示
        pbar.set_postfix({'最佳损失': f'{intermediate_result.fun:.4f}'})

    result = differential_evolution(
        func=wrapped_objective_func,
        bounds=active_bounds,
        x0=initial_active_params,
        maxiter=max_generations,
        popsize=popsize_multiplier,
        mutation=(0.5, 1.0),
        recombination=0.7,
        strategy='best1bin',
        disp=False,
        callback=callback,
        workers=-1
    )
    pbar.close()

    # =========================================================================
    #  ✅ Step 6: 优化后结果打印、保存与可视化
    # =========================================================================
    
    # --- a. 重构完整的最终参数向量 ---
    final_full_params = full_initial_params_vec.copy()

    if ENFORCE_SYMMETRY and NUM_UP_CONTROL_POINTS % 2 != 0 and NUM_DOWN_CONTROL_POINTS % 2 != 0:
        # result.x 是优化后的独立活动参数短向量
        temp_unique_params = initial_unique_params.copy()
        temp_unique_params[active_unique_mask] = result.x
        
        unique_up_res, unique_down_res, light_res = np.split(temp_unique_params, [num_unique_up, num_unique_up + num_unique_down])
        
        final_up_offsets = np.concatenate([unique_up_res, np.flip(unique_up_res[:-1])])
        final_down_offsets = np.concatenate([unique_down_res, np.flip(unique_down_res[:-1])])
        final_light_params = light_res
        
        final_full_params = np.concatenate([final_up_offsets, final_down_offsets, final_light_params])
    else:
        final_full_params[full_active_mask] = result.x

    # --- b. 从最终的偏移量计算出最终的绝对Y坐标 ---
    final_up_offsets, final_down_offsets, final_light_params = np.split(final_full_params, [NUM_UP_CONTROL_POINTS, NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS])
    final_up_y_coords = base_up_y - final_up_offsets
    final_down_y_coords = base_down_y + final_down_offsets
    
    # --- c. 保存本次优化的最终【绝对曲线坐标】和维度信息 ---
    output_dir = os.path.dirname(OUTPUT_PARAMS_FILE)
    if not os.path.exists(output_dir) and output_dir:
        os.makedirs(output_dir)
        
    np.savez_compressed(
        OUTPUT_PARAMS_FILE,
        up_y_coords=final_up_y_coords,
        down_y_coords=final_down_y_coords,
        light_params=final_light_params,
        num_up=NUM_UP_CONTROL_POINTS,
        num_down=NUM_DOWN_CONTROL_POINTS
    )

    final_up_offsets, final_down_offsets, final_light_params = np.split(final_full_params, [NUM_UP_CONTROL_POINTS, NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS])
    
    print("\n==============================================")
    print("        优化完成 (Optimization Complete)") 
    print("==============================================")    
    print(f"\n[+] 最佳损失值 (Lowest Loss): {result.fun:.6f}") 
    print("\n[+] 最佳参数 (Optimized Parameters):")    
    print(f" - 上表面偏移量: {np.round(final_up_offsets, 4)}")    
    print(f" - 下表面偏移量: {np.round(final_down_offsets, 4)}")  
    print(f" - 光源参数: {np.round(final_light_params, 4)}")  
    print("\n" + "="*46)   
    print("\nStep 4: 正在生成优化后的场景可视化...")   
    fig_after, ax_after = plt.subplots(figsize=(12, 12))   

    final_device_params = {    
        'up_y_coords': final_up_y_coords,   
        'down_y_coords': final_down_y_coords,   
        'bound': DEVICE_X_BOUNDS,  
        'num_control_points_up': NUM_UP_CONTROL_POINTS,    
        'num_control_points_down': NUM_DOWN_CONTROL_POINTS 
    }  
    plot_title = ( 
        f"After Optimize\n"    
        f"loss: {result.fun:.4f}"    
    )  
    setup_and_visualize(ax_after, plot_title, final_device_params, final_light_params, evaluator)  
    plt.show()