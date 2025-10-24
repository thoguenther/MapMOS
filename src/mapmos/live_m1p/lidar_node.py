from time import time

from eato_base import EATONode, EATOTopics, ConfigLoader
from robosense_api import RSLidarClient

from mapmos.live_m1p.umfeldconfig import UmfeldNodeConfig

class LidarNode(EATONode):
    def __init__(
        self,
        lidar_id: str,
    ):
        super().__init__()
        self.lidar_id = lidar_id
        self.lidar_client = None

    def load_config(self, config_loader: ConfigLoader):
        self.custom_config = config_loader.get("umfeld", UmfeldNodeConfig)

    def register_topics(self, eato_topics: EATOTopics):
        self.point_cloud_publisher = self.get_publisher(eato_topics.LIDAR_POINT_CLOUD_XYZI(self.lidar_id))

    def setup(self):
        self.lidar_client = RSLidarClient(
            lidar_type= self.custom_config.sensor_setup[self.lidar_id].type,
            pcap_path=None,
            group_address= self.custom_config.sensor_setup[self.lidar_id].group_address,
            host_address=self.custom_config.sensor_setup[self.lidar_id].host_address,
            pcap_repeat=False,
            point_cloud_size=self.eato_config.sensor_setup[self.lidar_id].data_shape[0],
        )
        if not self.lidar_client.open():
            self.logger.info(f"Error initializing LiDAR {self.lidar_id}. Shutting down")
            self.shutdown()
            

    def run_loop(self):
        point_cloud = self.lidar_client.get(timeout=1.0) # timeout in sec
        timestamp: float = time()
        if point_cloud is not None:
            array = point_cloud.numpy()
            self.point_cloud_publisher.write(array, timestamp=timestamp)
        else:
            self.logger.warn("Timeout waiting for point cloud")

    def shutdown(self):
        self.logger.info("Stopping lidar client")
        self.lidar_client.close()
