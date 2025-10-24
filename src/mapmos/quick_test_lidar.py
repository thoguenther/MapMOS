# quick_test_lidar.py
from mapmos.live_m1p.lidar_new import Lidar_pipeline
lidar = Lidar_pipeline()
arr = lidar.get_array()
print("frame:", None if arr is None else (arr.shape, arr.dtype))