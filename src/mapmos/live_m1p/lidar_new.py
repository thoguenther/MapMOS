import logging
import time
from robosense_api import RSLidarClient  # Beispiel, wie gehabt
import open3d as o3d
import numpy as np
import asyncio
from multiprocessing.synchronize import Event as EventType



class LidarPipeline:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        self.lidar_client = RSLidarClient(
            lidar_type="RSM1",
            pcap_path=None,
            group_address="0.0.0.0",
            host_address="192.168.1.102",
            pcap_repeat=False,
            point_cloud_size=78750,
        )
        # # a = self.get_array(max_wait_s=5.0)
        # # print(a)

    def get_array(self, max_wait_s: float = 5.0):
        start = self.lidar_client.open()
        if not start:
            self.lidar_client.close()
            raise SystemExit("Error setting up LiDAR.")
        deadline = time.monotonic() + max_wait_s   # <— einmal vor der Schleife!
        try:
            while time.monotonic() < deadline:
                
                point_cloud = self.lidar_client.get(timeout=0.1)

                if point_cloud is not None:
                    return point_cloud.numpy()

                # Falls die API bei Timeout None zurückgibt:
                self.logger.warning("Timeout waiting for point cloud (None received)")

            # Wenn wir hier ankommen: Gesamtwartezeit überschritten
            raise TimeoutError(f"No point cloud received within {max_wait_s:.1f}s")

        finally:
            self.lidar_client.close()

    def save_point_clouds(self, count=5, output_dir="./pointclouds"):
        """Speichert mehrere Punktwolken als .ply-Dateien"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        for i in range(count):
            print(f"→ Hole Punktwolke {i+1}/{count} …")
            points = self.get_array(max_wait_s=5.0)

            if points.shape[1] < 3:
                raise ValueError("Point cloud array muss mindestens (N,3) sein!")

            # open3d-Objekt erstellen
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            # optional: falls Intensität existiert → Grauwertfarbe
            if points.shape[1] >= 4:
                intensities = points[:, 3]
                colors = np.tile(intensities.reshape(-1, 1), (1, 3))
                colors /= np.max(colors) if np.max(colors) > 0 else 1
                pcd.colors = o3d.utility.Vector3dVector(colors)

            filename = os.path.join(output_dir, f"cloud_{i:03d}.ply")
            o3d.io.write_point_cloud(filename, pcd)
            print(f"✅ Gespeichert: {filename}")


# --- Lidar Client Wrapper ---
class RobosenseClientWrapper:
    def __init__(self, lidar_type: str = "RSM1", host_address: str = "192.168.1.102"):
        self.client = RSLidarClient(
            lidar_type=lidar_type,
            pcap_path=None,
            group_address="0.0.0.0",
            host_address=host_address,
            pcap_repeat=False,
            point_cloud_size=78750,
            max_queue_size=1
        )
        self.is_running = False

    def start(self):
        if not self.client.open():
            print("❌ Error initializing LiDAR.")
            self.is_running = False
            return False
        self.is_running = True
        print("✅ Lidar client initialized successfully.")
        return True

    def stop(self):
        if self.is_running:
            self.client.close()
        self.is_running = False

    def get_point_cloud_numpy(self, timeout: float = 0.1) -> np.ndarray | None:
        if not self.is_running:
            return None
        point_cloud = self.client.get(timeout=timeout)
        if point_cloud is not None:
            return point_cloud.numpy()
        return None
    
async def lidar_worker_async(
        lidar_id: str,
        # path: Path,
        shutdown_event: EventType
        ):
    # lid_output_path = path / "lidars" / lidar_id
    # lid_output_path.mkdir(parents=True, exist_ok=True)

    client = RSLidarClient(
            lidar_type="RSM1",
            group_address="0.0.0.0",
            host_address="192.168.1.102",
            point_cloud_size=78750,
        )
    pending_writes: list[tuple[asyncio.Task, float]] = []

    try:
        if not client.open():
            return
        print(f"✅\tLidar {lidar_id} initialized and started.")
        while not shutdown_event.is_set():
            # Run the blocking get() call in a separate thread
            point_cloud = await asyncio.to_thread(client.get, timeout=0.1)

            # if point_cloud is not None:
                # file_path = lid_output_path / f"{time.time_ns()}.npy"
                # write_task = asyncio.create_task(save_npy_async(file_path, point_cloud))
                # pending_writes.append((write_task, time.monotonic()))

            # Check for timed-out writes
            remaining_writes = []
            for task, creation_time in pending_writes:
                if time.monotonic() - creation_time > 5.0:
                    try:
                        await asyncio.wait_for(task, timeout=0.1)
                    except asyncio.TimeoutError:
                        print(f"❌\tLidar worker error ({lidar_id}): A file write operation timed out after 5 seconds.", file=sys.stderr)
                        task.cancel()
                        shutdown_event.set()
                elif not task.done():
                    remaining_writes.append((task, creation_time))
            pending_writes = remaining_writes
    finally:
        print(f"ℹ️\tShutting down lidar ({lidar_id}). Waiting for final file writes to complete...")
        if client:
            client.close()
        # Wait for any final writes to complete
        final_tasks = [task for task, _ in pending_writes if not task.done()]
        if final_tasks:
            await asyncio.gather(*final_tasks)
        print(f"✅\tLidar ({lidar_id}) process finished.")