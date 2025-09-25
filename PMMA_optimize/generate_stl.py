from utils.utils_ray_trace import generate_device_stl_from_npz
# --- 2. 调用核心函数 ---
output_stl_file = 'PMMA_optimize/output/侧壁_optimization_result_3_0.stl'
demo_npz_file = 'PMMA_optimize/output/侧壁_optimization_result_3_0.npz'
device_x_range = (0, 15)
device_y_range = (0, 10) # 定义器件的宽度

# 运行生成器
light_positions = generate_device_stl_from_npz(
    npz_path=demo_npz_file,
    stl_path=output_stl_file,
    x_range=device_x_range,
    y_range=device_y_range,
    resolution=100 # 提高分辨率可以使模型更平滑
)

# --- 3. 打印最终结果 ---
if light_positions:
    print("\n--- 最终结果 ---")
    print("计算出的光源在世界坐标系下的三维坐标 (x, y, z):")
    for i, pos in enumerate(light_positions):
        print(f"  光源 {i+1}: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")