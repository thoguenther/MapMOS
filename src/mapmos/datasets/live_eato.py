

# src/mapmos/live_inference_node.py
from __future__ import annotations
from typing import Any, Dict, Optional, Sequence
import time
from pathlib import Path
import numpy as np
from eato_base import EATONode, EATOTopics, ConfigLoader



class LiveEatoDataset(EATONode):
    """Dataset-Interface im Stil von rosbag.py / dateibasierten Quellen.

    - Hat __len__/__getitem__ (für die Pipeline).
    - __getitem__(i) holt "den nächsten" Frame (Index wird ignoriert).
    - sequences(): liefert ["live"], damit die CLI 'mapmos_pipeline -s live' akzeptiert.
    """
    # Hilft der CLI ggf. zu erkennen, dass -s optional ist:
    requires_sequence = False

    def __init__(self,
                #  root: str,                      # ungenutzt (Kompatibilität)
                 data_dir: Sequence[Path],
                 topic: str,
                 sequence: Optional[str]=None,   # CLI setzt hier z.B. "live"
                 lidar_id: str="M1P",
                 use_external_poses: bool=False,
                 max_frames: Optional[int]=None,
                 block_timeout: float=0.2,
                 *_, **__):
        super().__init__()
        self.data_dir = data_dir
        self.lidar_id = lidar_id
        self.subscribers = None
        self._use_external_poses = use_external_poses
        self._max_frames = max_frames
        self._block_timeout = block_timeout

        # # Tap starten (wie in deiner LiveViewNode)
        # self.setup()

        # Optional: zweite Quelle für Posen
        self._pose_sub = None  # self.get_subscriber(...), falls vorhanden

        # interner Zähler für __len__
        self._seen = 0

    def __len__(self) -> int:
        # Live hat per se keine Länge; gib entweder max_frames an
        if self._max_frames is not None:
            return self._max_frames
        # oder eine große Zahl, damit die Pipeline happy ist:
        return 2**31 - 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """MapMOS erwartet ein Dict mit mindestens:
           - 'points': (N,3|4) float32
           - 'stamp': float (Sekunden)
           optional:
           - 'pose': (4,4) float64
        """
        # Lies einen Frame (blockiert kurz)
        r = self.read_once(wait_new=True, timeout=self._block_timeout)
        if not getattr(r, "success", False) or r.value is None:
            # Bei Timeout / kein neuer Frame – versuche nochmal
            # (oder wirf eine Exception, je nach gewünschtem Verhalten)
            raise IndexError("No live frame available")  # triggert erneuten Versuch durch DataLoader

        pts = r.value
        if not isinstance(pts, np.ndarray) or pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("LiveEatoDataset: unexpected point array shape")

        pts = pts.astype(np.float32, copy=False)
        if pts.shape[1] >= 4:
            pts = pts[:, :4]  # XYZI
        else:
            pts = pts[:, :3]  # XYZ

        stamp = float(getattr(r, "timestamp", time.time()))

        sample: Dict[str, Any] = {
            "points": pts,
            "stamp": stamp,
        }

        if self._use_external_poses and self._pose_sub is not None:
            pr = self._pose_sub.read(wait_new=False, timeout=0.0)
            if getattr(pr, "success", False) and pr.value is not None:
                pose = np.asarray(pr.value, dtype=np.float64)
                if pose.shape == (4,4):
                    sample["pose"] = pose

        self._seen += 1
        return sample

    def load_config(self, config_loader: ConfigLoader):
        # falls du UmfeldConfig brauchst:
        # self.umfeld = config_loader.get("umfeld", UmfeldNodeConfig)
        # self.gain_str = f"{self.custom_config.sensor_setup[self.sensors[0]].gain}"
        # self.exposure_str = f"{self.custom_config.sensor_setup[self.sensors[0]].exposure}"
        pass
    
    # def register_topics(self, eato_topics: EATOTopics):
    #     topic = eato_topics.LIDAR_POINT_CLOUD_XYZI(self.lidar_id)
    #     self.subscribers = self.get_subscriber(topic)

    def register_topics(self, eato_topics: EATOTopics):
        self.subscribers.clear()
        self.logger.success(f"register_topics")
        data_subs: Dict[str, Any] = {}
        topic = eato_topics.LIDAR_POINT_CLOUD_XYZI(self.lidar_id)
        if not topic:
            self.logger.error(f"Unsupported or unmapped sensor '{self.lidar_id}'.")
        data_id = topic.topic_id.split("/")[-1]
        data_subs[data_id] = self.get_subscriber(topic)

        if not data_subs:
            self.logger.warn(f"No topics resolved for sensor M1P.")

            self.subscribers = data_subs
    
    # def setup(self):
    #     # self.setup() / self.load_config(...) / self.register_topics(...)
    #     # Beispiel-Pattern aus deiner LiveViewNode:
    #     # topic = EATOTopics.LIDAR_POINT_CLOUD_XYZI(self.lidar_id)
    #     # self.sub = self.get_subscriber(topic)
    #     pass

    def setup(self):
        self.set_target_rate(30.0)  # UI-friendly loop
        self.logger.success(f"setup")
        # Prepare per-stream timers
        for sensor_id, data_subs in self.subscribers.items():
            for data_id in data_subs.keys():
                kind = "RGB" if data_id in ("rgb", "raw") else "LiDAR"
                key = (sensor_id, kind)
                self._last_ts[key] = None
                self._fps[key] = None
    
    def shutdown(self):
        pass
    
    def run_loop(self):
        pass  # MapMOS zieht, wir pushen nicht

    # ---- CLI-Kompatibilität: welche Sequenzen gibt es? ----
    @staticmethod
    def sequences(root: str) -> list[str]:
        # Liefere wenigstens eine Pseudo-Sequenz, damit -s nicht zwingend ist
        return ["live"]
    
    def read_once(self, wait_new: bool=True, timeout: float=0.1):
        """Liest genau einen Frame. Ersetze das durch deinen echten Subscriber-Aufruf.
           Erwartet Rückgabe mit .success, .value (np.ndarray Nx3/4), .timestamp
        """
        class Res:
            success=False; value=None; timestamp=None
        r = Res()
        # --- DEMO: hier echten read() nutzen ---
        print(self.subscribers)
        rr = self.subscribers.read(wait_new=wait_new, timeout=timeout)
        return rr
        # return r
    
    
    

    