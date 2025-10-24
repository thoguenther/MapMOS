# src/mapmos/datasets/live_m1p.py
import time
import threading
import queue
from typing import Dict, Any, Iterator, List, Optional

import numpy as np

# dein Wrapper (Pfad ggf. anpassen)
from mapmos.live_m1p.lidar_new import LidarPipeline
from pathlib import Path


class LiveM1PDataset:
    """
    Live-Quelle für MapMOS (Robosense M1P via Lidar_pipeline).

    Dieses Dataset ist "sequenzlos", stellt aber eine Pseudo-Sequenz 'live' bereit,
    damit die CLI zufrieden ist (-s live). Es liefert endlos Frames.

    Jeder Frame ist ein Dict mit:
      - 'points': np.ndarray (N,3|4) float32
      - 'stamp' : float (epoch seconds)
      - optional 'pose': np.ndarray (4,4) float64
    """
    # Manche Pipelines prüfen das:
    requires_sequence = False

    def __init__(self, data_dir, sequence: Optional[str] = None,
                 min_points: int = 1000, use_external_poses: bool = False,
                 **kwargs):
        self.data_dir = Path(data_dir)
        self.sequence = sequence or "live"
        self.min_points = int(min_points)
        self.use_external_poses = bool(use_external_poses)

        # Live-LiDAR
        self._lidar = LidarPipeline()

        # kleiner Puffer, damit die Pipeline nicht blockiert
        self._q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._producer, daemon=True)
        self._th.start()

    # ---- API, die viele PRBonn-Pipelines nutzen ----
    @staticmethod
    def sequences() -> List[str]:
        # erlaubt z. B. --sequence live
        return ["live"]

    def frames(self, sequence: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        # häufige API: liefert Iterator über Frames einer Sequenz
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.5)
                print("[live_m1p] yield frame:", item["points"].shape)
            except queue.Empty:
                print("[live_m1p] waiting for frames...")
                continue
            yield item

    # Fallbacks, falls andere Stellen darauf zugreifen:
    def __len__(self) -> int:
        # pseudo-unendlich
        return 10**12

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return self.frames(self.sequence)

    # ---- interner Producer-Thread ----
    def _producer(self):
        while not self._stop.is_set():
            arr = self._lidar.get_array()  # np.ndarray (N,C), C>=4 (x,y,z,i)
            if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 2 or arr.shape[1] < 3:
                print("[live_m1p] get_array() -> None (timeout?) or dim or shape... Zeile 71")
                continue
            print("[live_m1p] raw frame:", arr.shape, arr.dtype)

            pts = arr[:, :4].astype(np.float32, copy=False) if arr.shape[1] >= 4 else arr[:, :3].astype(np.float32, copy=False)

            # simple sanity
            good = np.isfinite(pts).all(axis=1)
            pts = pts[good]
            if pts.shape[0] < self.min_points:
                print(f"[live_m1p] too few points after filter: {pts.shape[0]}")
                continue

            stamp = time.time()
            item: Dict[str, Any] = {"points": pts, "stamp": stamp}
            # Wenn du externe Posen hast, hier ergänzen:
            # item["pose"] = T_4x4_float64

            try:
                # dropp, wenn voll (kein Backpressure ins LiDAR)
                self._q.put(item, timeout=0.01)
                print("[live_m1p] queued frame:", pts.shape)
            except queue.Full:
                print("[live_m1p] queue full -> drop")
                pass

    def shutdown(self):
        self._stop.set()
        try:
            self._th.join(timeout=0.5)
        except Exception:
            pass
