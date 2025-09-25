运行PMMA_optimize中optimize.py即可

通过分支来控制优化对象，分别为底面和侧壁

优化底面时，主要修改对象如下:

# --- a. 优化样条曲线精度、孔径、保存路径 ---
NUM_UP_CONTROL_POINTS = 7  
NUM_DOWN_CONTROL_POINTS = 7 
DEVICE_X_BOUNDS = [0, 16]

PARAMS_FILE = 'PMMA_optimize/output/0925/bottom_3_lights_116_7_0.npz' 
OUTPUT_PARAMS_FILE = f'PMMA_optimize/output/0925/bottom_3_lights_116_{NUM_UP_CONTROL_POINTS}_0.npz'

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

# 定义基准线
base_down_y = 3.65 #下表面初始形状
control_x_up = np.linspace(DEVICE_X_BOUNDS[0], DEVICE_X_BOUNDS[1], NUM_UP_CONTROL_POINTS)
base_up_y = 16 # 上表面初始形状

优化侧壁时，主要修改对象如下:

