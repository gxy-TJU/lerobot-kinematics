import cv2
import numpy as np
def pixel2world(x, y):
    # 内参矩阵和畸变系数
    K = np.array([[474.1750, 0, 0], [0, 476.5389, 0], [320.5032, 240.9307, 1]])  # 相机内参矩阵
    dist_coeffs = np.array([0.1935, -0.3276, 0, 0, 0.2103])  # 畸变系数

    # 像素坐标
    pixel_coords = np.array([x, y], dtype=np.float32)
    
    # 去畸变
    undistorted_coords = cv2.undistortPoints(np.expand_dims(pixel_coords, axis=0), K, dist_coeffs)
    
    #print(undistorted_coords)
    # 假设深度值为 depth
    depth = 319  # 示例深度值

    # 反投影到相机坐标系
    undistorted_coords_homogeneous = np.append(undistorted_coords, 1)  # 齐次坐标
    camera_coords = np.linalg.inv(K) @ undistorted_coords_homogeneous * depth

    # 外参矩阵：旋转矩阵 R 和平移向量 T
    R = np.array([[-0.0129, 0.9969, -0.0770], [-0.9999, -0.0125, 0.0053], [0.0043, 0.0771, 0.9970]])  # 旋转矩阵
    T = np.array([81.2526, -117.0128, 319.3409])  # 平移向量

    # 从相机坐标系到世界坐标系的转换
    world_coords = R @ camera_coords + T
    return world_coords

if __name__ == "__main__":
    x, y = 320, 240
    world_coords = pixel2world(x, y)
    print(world_coords)
