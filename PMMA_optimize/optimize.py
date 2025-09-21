import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.collections import LineCollection
import time
import os
from scipy.interpolate import CubicSpline
from utils.utils_ray_trace import LC_device, Point_light_source
from utils.utils_lossEvaluator import PlanarLossEvaluator, calculate_ray_line_intersections
from utils.utils_free_form_prism_design import visualize_scene
from scipy.optimize import differential_evolution



# =============================================================================
# 3. 动态光源生成与优化目标封装 (2D版本)
# =============================================================================
def calculate_light_sources_from_params(a, l1, l2, l3):
    """
    【2D版本】根据斜率a和三个相对位置参数l1,l2,l3，计算三个光源的位置和姿态。

    参数定义已更新:
    :param a: 直线的斜率 (y = a*x + c)
    :param l1: 中间光源在整个线段 (p_start 到 p_end) 上的位置比例。范围[-1, 1]。
              -1代表p_start, 0代表中心点, 1代表p_end。
    :param l2: 左侧光源在"中间光源"与"左端点(p_start)"所构成线段上的位置比例。范围[0, 1]。
              0代表与中间光源重合, 1代表与左端点重合。
    :param l3: 右侧光源在"中间光源"与"右端点(p_end)"所构成线段上的位置比例。范围[0, 1]。
              0代表与中间光源重合, 1代表与右端点重合。
    :return: light_sources_config_2d 列表
    """
    # --- a. 计算直线与矩形的交点 (此部分逻辑不变) ---
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

    unique_intersections = np.unique(np.round(intersections, decimals=5), axis=0)
    if len(unique_intersections) != 2:
        return []
    
    p_start, p_end = unique_intersections
    # 为清晰起见，确保 p_start 的 x 坐标更小
    if p_start[0] > p_end[0]:
        p_start, p_end = p_end, p_start
    
    # --- b. 计算线段中心和半长向量 (此部分逻辑不变) ---
    segment_center = (p_start + p_end) / 2.0
    segment_half_vector = (p_end - p_start) / 2.0

    # ####################################################################
    # ## c. 计算光源位置 (已按新逻辑重写) ##
    # ####################################################################
    
    # 1. 根据 l1 计算中间光源的位置
    pos_middle = segment_center + l1 * segment_half_vector
    
     # 计算朝向左端点(p_start)的单位向量
    vec_to_start = p_start - pos_middle
    norm_start = np.linalg.norm(vec_to_start)
    # 如果中间光源与端点重合，则锚点也与端点重合，避免除零错误
    if norm_start < 1e-9:
        p_left_anchor = pos_middle
    else:
        unit_vec_to_start = vec_to_start / norm_start
        p_left_anchor = pos_middle + 1.0 * unit_vec_to_start

    # 计算朝向右端点(p_end)的单位向量
    vec_to_end = p_end - pos_middle
    norm_end = np.linalg.norm(vec_to_end)
    if norm_end < 1e-9:
        p_right_anchor = pos_middle
    else:
        unit_vec_to_end = vec_to_end / norm_end
        p_right_anchor = pos_middle + 1.0 * unit_vec_to_end

    # 3. 根据 l2，在新的“左锚点”和“左端点(p_start)”之间进行线性插值
    pos_left = p_left_anchor + l2 * (p_start - p_left_anchor)
    
    # 4. 根据 l3，在新的“右锚点”和“右端点(p_end)”之间进行线性插值
    pos_right = p_right_anchor + l3 * (p_end - p_right_anchor)
    
    # 5. 组合并塑形
    positions_2d = [pos_left, pos_middle, pos_right]
    positions_2d_col = [pos.reshape(2, 1) for pos in positions_2d]
    
    # 4. 组合并塑形
    positions_2d = [pos_left, pos_middle, pos_right]
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
# =============================================================================
#                         --- 主执行流程 ---
# =============================================================================
if __name__ == '__main__':
    
    # =========================================================================
    #  ✅ Step 1: 配置区 (在此处修改所有参数)
    # =========================================================================
    print("Step 1: 正在设置优化问题的配置...")

    # --- a. 文件与几何配置 ---
    NUM_UP_CONTROL_POINTS = 50  # <-- 您可以修改这里的数量来进行维度扩展
    NUM_DOWN_CONTROL_POINTS = 50
    DEVICE_X_BOUNDS = [0, 15.2]
    
    # 定义用于加载和保存的文件名
    PARAMS_FILE = 'PMMA_optimize/output/0922/optimization_result_3_1.npz' 
    OUTPUT_PARAMS_FILE = f'PMMA_optimize/output/0922/optimization_result_{NUM_UP_CONTROL_POINTS}_1.npz'
    
    # ... 其他配置保持不变 ...
    initial_light_params_defaults = [-0.1, 0, 0.5, 0.5]
    evaluator_config = {
        'line_normal': [1, 0], 'line_center': [0, 29], 'line_length': 20.0,
        'weights': [0.8, 0.1, 0.1]
    }
    constraints_cfg = {'x_range': DEVICE_X_BOUNDS, 'penalty_weight': 1000.0}
    
    # --- c. 定义 *所有* 参数的完整边界 ---
    up_offset_bounds = [(0, 1)] * NUM_UP_CONTROL_POINTS      # 偏移量的搜索范围可以设置得小一些
    down_offset_bounds = [(0, 1)] * NUM_DOWN_CONTROL_POINTS
    light_bounds = [(-0.2, 0), (-0.33, 0.33), (0, 1),  (0, 1)]
    full_bounds = up_offset_bounds + down_offset_bounds + light_bounds

    # --- d. 参数冻结配置 ---
    up_active_mask = [True] * NUM_UP_CONTROL_POINTS
    down_active_mask = [True] * NUM_DOWN_CONTROL_POINTS
    light_active_mask = [True, True, True, True]
    active_params_mask = np.array(up_active_mask + down_active_mask + light_active_mask)
    
    # --- e. 优化器超参数 ---
    max_generations = 1000
    popsize_multiplier = 20
    
    # =========================================================================
    #  ✅ Step 2: 初始化参数 (新功能：加载基准线 或 默认生成)
    # =========================================================================
    print("\nStep 2: 正在初始化优化起点...")
    
    # 【新逻辑】定义当前运行的基准线，它可能来自文件，也可能来自默认值
    if os.path.exists(PARAMS_FILE):
        try:
            print(f"--> 成功从 '{PARAMS_FILE}' 加载上次的优化结果作为新基准。")
            data = np.load(PARAMS_FILE)
            loaded_up_y = data['up_y_coords']
            loaded_down_y = data['down_y_coords']
            loaded_num_up = data['num_up']
            loaded_num_down = data['num_down']

            # 【新逻辑】如果控制点数量发生变化，则通过插值扩展基准线
            if loaded_num_up != NUM_UP_CONTROL_POINTS:
                print(f"--> 上表面维度不匹配，正在从 {loaded_num_up} 点插值为 {NUM_UP_CONTROL_POINTS} 点...")
                old_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], loaded_num_up)
                temp_spline = CubicSpline(old_control_x, loaded_up_y)
                new_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
                base_up_y = temp_spline(new_control_x) # 插值生成新的基准线
            else:
                base_up_y = loaded_up_y # 维度未变，直接使用

            if loaded_num_down != NUM_DOWN_CONTROL_POINTS:
                print(f"--> 下表面维度不匹配，正在从 {loaded_num_down} 点插值为 {NUM_DOWN_CONTROL_POINTS} 点...")
                old_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], loaded_num_down)
                temp_spline = CubicSpline(old_control_x, loaded_down_y)
                new_control_x = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_DOWN_CONTROL_POINTS)
                base_down_y = temp_spline(new_control_x)
            else:
                base_down_y = loaded_down_y

            # 光源参数直接继承
            initial_light_params = data.get('light_params', initial_light_params_defaults)
            
        except Exception as e:
            print(f"--> 加载文件 '{PARAMS_FILE}' 出错: {e}。将使用默认起点。")
            # 出错则回退到默认值
            control_x_up = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
            base_up_y = 39.54 - control_x_up
            base_down_y = np.zeros(NUM_DOWN_CONTROL_POINTS)
            initial_light_params = initial_light_params_defaults

    else:
        print(f"--> 未找到 '{PARAMS_FILE}'。将使用默认的硬编码基准线作为起点。")
        control_x_up = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
        base_up_y = 39.54 - control_x_up
        base_down_y = np.zeros(NUM_DOWN_CONTROL_POINTS)
        initial_light_params = initial_light_params_defaults

    # 【新逻辑】无论起点如何，本轮优化的初始偏移量都清零
    initial_up_offsets = np.zeros(NUM_UP_CONTROL_POINTS)
    initial_down_offsets = np.zeros(NUM_DOWN_CONTROL_POINTS)
    
    # 完整的初始参数向量（优化器看到的）总是由零偏移量和光源参数构成
    full_initial_params_vec = np.concatenate([initial_up_offsets, initial_down_offsets, initial_light_params])
    
    # =========================================================================
    #  ✅ Step 3: 准备优化 (处理冻结参数等)
    # =========================================================================
    
    active_bounds = [b for i, b in enumerate(full_bounds) if active_params_mask[i]]
    initial_active_params = full_initial_params_vec[active_params_mask]

    # 实例化核心组件
    evaluator = PlanarLossEvaluator(**evaluator_config)
    device_static_config = {
        'num_up_control_points': NUM_UP_CONTROL_POINTS,
        'num_down_control_points': NUM_DOWN_CONTROL_POINTS,
        'bound': DEVICE_X_BOUNDS,
        'base_up_y_line': base_up_y, # <-- 传入本轮的最终基准线
        'base_down_y_line': base_down_y,
    }
    objective_func = Objective(LC_device, Point_light_source, evaluator, device_static_config, constraints_cfg)

    def wrapped_objective_func(active_params):
        full_params = full_initial_params_vec.copy()
        full_params[active_params_mask] = active_params
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
    setup_and_visualize(ax_before, "Before Optimize", initial_device_params, light_params, evaluator)
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
    #  Step 6: 优化后结果打印、保存与可视化
    # =========================================================================
    
    # --- a. 重构完整的最终偏移量参数向量 ---
    final_full_params = full_initial_params_vec.copy()
    final_full_params[active_params_mask] = result.x

    # --- b. 从最终的偏移量计算出最终的绝对Y坐标 ---
    final_up_offsets, final_down_offsets, final_light_params = np.split(final_full_params, [NUM_UP_CONTROL_POINTS, NUM_UP_CONTROL_POINTS + NUM_DOWN_CONTROL_POINTS])
    
    final_up_y_coords = base_up_y - final_up_offsets
    final_down_y_coords = base_down_y + final_down_offsets
    
    # --- c. 【新逻辑】保存本次优化的最终【绝对曲线坐标】和维度信息 ---
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