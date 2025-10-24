# MIT License
#
# Copyright (c) 2023 Benedikt Mersch, Tiziano Guadagnino,
# Ignacio Vizzo, Cyrill Stachniss
#
# Modified 2025 for live LiDAR streaming (no dataset dependency)

from imp import lock_held
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from tqdm.auto import trange

from mapmos.config import load_config
from mapmos.mapmos_net import MapMOSNet
from mapmos.mapping import VoxelHashMap
from mapmos.metrics import get_confusion_matrix
from mapmos.odometry import Odometry
from mapmos.utils.pipeline_results import MOSPipelineResults
from mapmos.utils.save import KITTIWriter, PlyWriter, StubWriter
from mapmos.utils.visualizer import MapMOSVisualizer, StubVisualizer
from typing import Tuple
import sys

# NEW: bring in the live LiDAR source
# from mapmos.live_m1p.lidar_new import LidarPipeline, RobosenseClientWrapper

import time
from robosense_api import RSLidarClient  # Beispiel, wie gehabt
import open3d as o3d
import numpy as np
import asyncio
from multiprocessing.synchronize import Event as EventType
import multiprocessing as mp




class MapMOSPipeline:
    """
    Live MapMOS pipeline using a LiDAR stream.

    Dataset support has been removed. We continuously fetch point clouds from
    the LiDAR and process them online. Optionally stop after `n_scans` frames.
    """

    def __init__(
        self,
        weights: Path,
        config: Optional[Path] = None,
        visualize: bool = False,
        save_ply: bool = False,
        save_kitti: bool = False,
        n_scans: int = -1,
        max_wait_s: float = 5.0,
    ):
        # Live LiDAR
        # self.lidar = LidarPipeline()
        # self.lidar = RobosenseClientWrapper()
        self.max_wait_s = max_wait_s

        # Config & output
        self.config = load_config(config)
        self.results_dir = None

        # Model
        state_dict = {
            k.replace("mos.", ""): v for k, v in torch.load(weights)["state_dict"].items()
        }
        self.model = MapMOSNet(self.config.mos.voxel_size_mos)
        self.model.load_state_dict(state_dict)
        self.model.cuda().eval().freeze()

        # Odometry (was provided by OdometryPipeline before)
        self.odometry = Odometry(self.config.data, self.config.odometry)

        # Belief map & delay buffer
        self.belief = VoxelHashMap(
            voxel_size=self.config.mos.voxel_size_belief,
            max_distance=self.config.mos.max_range_belief,
        )
        self.buffer = deque(maxlen=self.config.mos.delay_mos)

        # Results/stats
        self.results = MOSPipelineResults()
        self.n_scans = n_scans
        self.times_mos = []
        self.times_belief = []
        self.confusion_matrix_belief = torch.zeros(2, 2)
        self.has_gt = False  # live stream has no GT
        self.times_total = []
        self.fps_ema = None

        # Vis & writers
        self.visualize = visualize
        self.visualizer = MapMOSVisualizer() if visualize else StubVisualizer()
        self.visualizer.set_voxel_size(self.config.mos.voxel_size_belief)
        self.ply_writer = PlyWriter() if save_ply else StubWriter()
        self.kitti_writer = KITTIWriter() if save_kitti else StubWriter()

    # ---------------- Public API ----------------
    def run(self):
        """Run online until `n_scans` processed (or forever if -1)."""
        self._create_output_dir()
        shutdown_event = mp.Event()

        with torch.no_grad():
            asyncio.run(self._run_pipeline_live(shutdown_event=shutdown_event))
        # No GT → skip odom eval; still report MOS stats/fps collected
        print("Log schreiben")
        self._write_cfg()
        self._write_log()
        return self.results

    # ---------------- Internals ----------------
    def _create_output_dir(self):
        # Lightweight output dir for live mode
        root = Path(getattr(self.config, "results_dir", "results_live"))
        root.mkdir(parents=True, exist_ok=True)
        (root / "ply").mkdir(exist_ok=True)
        (root / "bin" / "sequences" / "live" / "predictions").mkdir(parents=True, exist_ok=True)
        self.results_dir = str(root)

    def _write_cfg(self):
        # Optional: dump the effective config
        try:
            cfg_path = Path(self.results_dir) / "config_used.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write(self.config.to_yaml())
        except Exception:
            pass

    def _write_log(self):
        # Aggregate and print/record FPS
        self.results.eval_fps(self.times_mos, desc="Average Frequency MOS")
        self.results.eval_fps(self.times_belief, desc="Average Frequency Belief")
        if self.times_total:
            self.results.eval_fps(self.times_total, desc="Average Frequency Total")
         # optional: CSV
        # try:
        #     import csv
        #     with open(Path(self.results_dir) / "timings.csv", "w", newline="", encoding="utf-8") as f:
        #         w = csv.writer(f)
        #         w.writerow(["frame_idx","mos_ns","belief_ns","total_ns","fps_inst","fps_ema"])
        #         fps_ema = None
        #         for i,(mos,bel,total) in enumerate(zip(
        #                 self.times_mos,
        #                 self.times_belief,
        #                 self.times_total[:len(self.times_mos)])):
        #             fps_inst = 1e9/total if total>0 else 0.0
        #             fps_ema = fps_inst if fps_ema is None else 0.9*fps_ema + 0.1*fps_inst
        #             w.writerow([i,mos,bel,total,fps_inst,fps_ema])
        # except Exception:
        #     pass

    def _preprocess(self, points, min_range, max_range):
        ranges = np.linalg.norm(points - self.odometry.current_location(), axis=1)
        mask = ranges <= max_range if max_range > 0 else np.ones_like(ranges, dtype=bool)
        mask = np.logical_and(mask, ranges >= min_range)
        return mask

    # def _fetch_scan(self):
    #     """Blocking fetch from LiDAR, returns (Nx3) points, timestamps (Nx1), gt (-1s)."""
    #     t0 = time.perf_counter()        
    #     # arr = self.lidar.get_array(max_wait_s=self.max_wait_s)
    #     # arr = self.lidar.get_point_cloud_numpy()
    #     t1 = time.perf_counter()
    #     print("get ptc ", t1 - t0)
    #     if arr is not None:
    #         if arr.shape[1] < 3:
    #             raise ValueError("LiDAR array must have at least 3 columns (x,y,z)")
    #         pts = arr[:, :3].astype(np.float64)
    #     ts = np.zeros((pts.shape[0],), dtype=np.float64)  # live: no per-point timestamps
    #     gt = -1 * np.ones((pts.shape[0],), dtype=np.int32)
    #     return pts, ts, gt

# Erwartet, dass EventType woanders als Union/Typalias definiert ist
# z.B.: EventType = Union[asyncio.Event, threading.Event]

    async def _run_pipeline_live(self, shutdown_event: "EventType"):
        processed = 0
        # Progress bar only if n_scans>0
        pbar_iter = range(self.n_scans) if self.n_scans and self.n_scans > 0 else iter(int, 1)
        if isinstance(pbar_iter, range):
            iterator = trange(self.n_scans, unit=" frames", dynamic_ncols=True)
        else:
            iterator = pbar_iter  # infinite iterator

        client = RSLidarClient(
            lidar_type="RSM1",
            group_address="0.0.0.0",
            host_address="192.168.1.102",
            point_cloud_size=78750,
        )

        # Liste (Task, Erstellungszeit) – analog zu lidar_worker_async
        pending_tasks: list[Tuple[asyncio.Task, float]] = []

        try:
            if not client.open():
                return
            print("✅\tLidar initialized and started.")

            for _ in iterator:
                if shutdown_event.is_set():
                    break

                t0 = time.perf_counter()

                # Blockierendes LiDAR-Get in Thread ausführen, Event Loop bleibt reaktiv
                local_scan = await asyncio.to_thread(client.get, timeout=0.1)
                if local_scan is None:
                    # Kein Frame – kurz weiter, aber auch Gelegenheit, Tasks zu prüfen
                    pass
                else:
                    local_scan = local_scan.numpy()[:, :3]
                    timestamps = np.zeros((local_scan.shape[0],), dtype=np.float64)  # live: no per-point timestamps
                    gt_labels = -1 * np.ones((local_scan.shape[0],), dtype=np.int32)

                    # Map points aus Odometry
                    map_points, map_indices = self.odometry.get_map_points()

                    # Registrierung (aktualisiert last_pose intern)
                    scan_points = self.odometry.register_points(local_scan, timestamps, processed)

                    # --- Pre-Filter (MOS Working Range) ---
                    min_range_mos = self.config.mos.min_range_mos
                    max_range_mos = self.config.mos.max_range_mos
                    scan_mask = self._preprocess(scan_points, min_range_mos, max_range_mos)
                    scan_points = torch.tensor(scan_points[scan_mask], dtype=torch.float32, device="cuda")
                    gt_labels = gt_labels[scan_mask]

                    map_mask = self._preprocess(map_points, min_range_mos, max_range_mos)
                    map_points = torch.tensor(map_points[map_mask], dtype=torch.float32, device="cuda")
                    map_indices = torch.tensor(map_indices[map_mask], dtype=torch.float32, device="cuda")

                    # --- MOS (GPU) ---
                    torch.cuda.synchronize()
                    pred_logits_scan, pred_logits_map = self.model.predict(
                        scan_points,
                        map_points,
                        processed * torch.ones(len(scan_points)).type_as(scan_points),
                        map_indices,
                    )
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    self.times_mos.append(t2 - t0)

                    # To CPU/NumPy
                    pred_logits_scan = pred_logits_scan.detach().cpu().numpy().astype(np.float64)
                    pred_logits_map  = pred_logits_map.detach().cpu().numpy().astype(np.float64)
                    scan_points_np   = scan_points.cpu().numpy().astype(np.float64)
                    map_points_np    = map_points.cpu().numpy().astype(np.float64)

                    pred_labels_scan = self.model.to_label(pred_logits_scan)
                    pred_labels_map  = self.model.to_label(pred_logits_map)

                    # --- Belief (CPU/Numpy) ---
                    map_mask_belief = pred_logits_map > 0
                    map_mask_belief = np.logical_and(
                        map_mask_belief,
                        self._preprocess(map_points_np, 0.0, self.config.mos.max_range_belief),
                    )
                    scan_mask_belief = self._preprocess(scan_points_np, 0.0, self.config.mos.max_range_belief)
                    points_stacked = np.vstack([scan_points_np[scan_mask_belief], map_points_np[map_mask_belief]])
                    logits_stacked = np.vstack(
                        [
                            pred_logits_scan[scan_mask_belief].reshape(-1, 1),
                            pred_logits_map[map_mask_belief].reshape(-1, 1),
                        ]
                    ).reshape(-1)

                    t3 = time.perf_counter()
                    self.belief.update_belief(points_stacked, logits_stacked)
                    self.belief.get_belief(scan_points_np)
                    t4 = time.perf_counter()
                    self.times_belief.append(t4 - t0)

                    # --- Visualization ---
                    t5 = time.perf_counter()
                    self.visualizer.update(
                        scan_points_np,
                        map_points_np,
                        pred_labels_scan,
                        pred_labels_map,
                        self.belief,
                        self.odometry.last_pose,
                    )
                    t6 = time.perf_counter()

                    # --- Delay-Buffer & (optionales) Speichern/Evaluieren ---
                    self.buffer.append([processed, scan_points_np, gt_labels])
                    if len(self.buffer) == self.buffer.maxlen:
                        query_index, query_points, query_labels = self.buffer.popleft()
                        # Analog zum File-Write im anderen Worker:
                        # _finalize_prediction blockiert potentiell (I/O); in Thread auslagern
                        task = asyncio.create_task(
                            asyncio.to_thread(self._finalize_prediction, query_index, query_points, query_labels)
                        )
                        pending_tasks.append((task, time.monotonic()))

                    # Housekeeping
                    self.belief.remove_voxels_far_from_location(self.odometry.current_location())

                    processed += 1

                    # --- Frame-Ende / FPS ---
                    t7 = time.perf_counter()
                    total_s = t7 - t0
                    self.times_total.append(total_s)
                    fps_inst = (1.0 / total_s) if total_s > 0 else 0.0
                    self.fps_ema = fps_inst if self.fps_ema is None else 0.9 * self.fps_ema + 0.1 * fps_inst
                    # if hasattr(iterator, "set_postfix"):
                    #     iterator.set_postfix(fps=f"{self.fps_ema:.1f}")

                    if self.n_scans > 0 and processed >= self.n_scans:
                        break

                # --- Timeout-Überwachung für Hintergrundtasks (analog zu lidar_worker_async) ---
                remaining: list[Tuple[asyncio.Task, float]] = []
                for task, created in pending_tasks:
                    # 5s Gesamtbudget pro Task
                    if time.monotonic() - created > 5.0:
                        try:
                            await asyncio.wait_for(task, timeout=0.1)
                        except asyncio.TimeoutError:
                            print(
                                "❌\tPipeline error: A finalize/eval task timed out after 5 seconds.",
                                file=sys.stderr,
                            )
                            task.cancel()
                            # Sofortige, kontrollierte Beendigung
                            shutdown_event.set()
                    elif not task.done():
                        remaining.append((task, created))
                pending_tasks = remaining

            # Schleifenende erreicht (Stop/Kontingent)
        finally:
            print("ℹ️\tShutting down live pipeline. Waiting for final tasks to complete...")
            try:
                if client:
                    client.close()
            except Exception as e:
                print(f"⚠️\tError while closing LiDAR client: {e}", file=sys.stderr)

            # Flush der restlichen Delay-Frames synchron (in Thread) oder als Tasks bündeln
            while len(self.buffer) != 0:
                query_index, query_points, query_labels = self.buffer.popleft()
                task = asyncio.create_task(
                    asyncio.to_thread(self._finalize_prediction, query_index, query_points, query_labels)
                )
                pending_tasks.append((task, time.monotonic()))

            # Ausstehende Tasks fertigstellen
            final_tasks = [t for t, _ in pending_tasks if not t.done()]
            if final_tasks:
                try:
                    await asyncio.gather(*final_tasks, return_exceptions=True)
                except Exception as e:
                    print(f"⚠️\tError while finalizing pending tasks: {e}", file=sys.stderr)

            print("✅\tLive pipeline finished.")


    def _finalize_prediction(self, query_index, query_points, query_labels):
        belief_query = self.belief.get_belief(query_points)
        belief_labels_query = self.model.to_label(belief_query)

        # Only compute confusion if GT available (not in live mode)
        if self.has_gt and query_labels is not None and (query_labels >= 0).any():
            self.confusion_matrix_belief += get_confusion_matrix(
                torch.tensor(belief_labels_query, dtype=torch.int32),
                torch.tensor(query_labels, dtype=torch.int32),
            )

        # Save
        self.ply_writer.write(
            query_points,
            belief_labels_query,
            query_labels,
            filename=f"{self.results_dir}/ply/{query_index:06}.ply",
        )
        self.kitti_writer.write(
            belief_labels_query,
            filename=f"{self.results_dir}/bin/sequences/live/predictions/{query_index:06}.label",
        )
