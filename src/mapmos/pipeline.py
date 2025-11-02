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
        shutdown_event = asyncio.Event()
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


    async def _run_pipeline_live(self, shutdown_event):
        client = RSLidarClient(
            lidar_type="RSM1",
            group_address="0.0.0.0",
            host_address="192.168.1.102",
            point_cloud_size=78750,
        )

        latest_frame = None           # <— gemeinsame Variable (ohne Queue)
        pending_tasks: list[Tuple[asyncio.Task, float]] = []
        processed = 0
        latest_ref = {"frame": None}

        async def producer():
            print("[producer] start", flush=True)
            try:
                if not client.open():
                    return
                print("✅\tLidar initialized and started.")

                while not shutdown_event.is_set():
                    # ~10 Hz vom Gerät (timeout 0.1s)
                    frame = await asyncio.to_thread(client.get, timeout=0.1)
                    if frame is not None:
                        latest_ref["frame"] = frame  # billiger Referenztausch

                        
                    else:
                        await asyncio.sleep(0)  # Loop responsiv halten
            except Exception as e:
                print(f"⚠️ producer error: {e}", file=sys.stderr)
                shutdown_event.set()  # globale Beendigung anstoßen




        async def consumer():
            # optional: Fortschrittsanzeige wie bisher
            print("[consumer] start", flush=True)            
    
            pts_xyz = np.zeros((78750, 3))

            try:
                while not shutdown_event.is_set():

                    # sauber getaktete Schleife mit Monotonic-Clock (kein Drift)
                    ## Aktuellen Schnappschuss holen; keine Versionierung
                    frame = latest_ref.get("frame")
                    if frame is not None:
                        # >>> Hier dein Processing <<<
                        # z. B.: self.process_frame(frame)
                        pts_xyz_old = pts_xyz
                        pts_xyz = frame.numpy()[:, :3]
                        if np.array_equal(pts_xyz, pts_xyz_old):
                            continue
                        finite_mask = np.isfinite(pts_xyz).all(axis=1)
                        pts_xyz = np.ascontiguousarray(pts_xyz[finite_mask], dtype=np.float64)
                        if pts_xyz.shape[0] == 0:
                            continue
                        timestamps = np.zeros((pts_xyz.shape[0],), dtype=np.float64)

                        t0 = time.perf_counter()

                        # Map / Odometry
                        map_points, map_indices = self.odometry.get_map_points()
                        scan_points = self.odometry.register_points(pts_xyz, timestamps, processed)

                        # Pre-Filter
                        min_r = self.config.mos.min_range_mos
                        max_r = self.config.mos.max_range_mos

                        scan_mask = self._preprocess(scan_points, min_r, max_r)
                        scan_points = torch.tensor(scan_points[scan_mask], dtype=torch.float32, device="cuda")
                        gt_labels = (-1 * np.ones((scan_mask.sum(),), dtype=np.int32))

                        map_mask = self._preprocess(map_points, min_r, max_r)
                        map_points = torch.tensor(map_points[map_mask], dtype=torch.float32, device="cuda")
                        map_indices = torch.tensor(map_indices[map_mask], dtype=torch.float32, device="cuda")

                        # MOS (GPU)
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

                        # Belief
                        map_mask_belief  = np.logical_and(pred_logits_map > 0,
                                            self._preprocess(map_points_np, 0.0, self.config.mos.max_range_belief))
                        scan_mask_belief = self._preprocess(scan_points_np, 0.0, self.config.mos.max_range_belief)
                        points_stacked   = np.vstack([scan_points_np[scan_mask_belief], map_points_np[map_mask_belief]])
                        logits_stacked   = np.vstack([pred_logits_scan[scan_mask_belief].reshape(-1,1),
                                                    pred_logits_map[map_mask_belief].reshape(-1,1)]).reshape(-1)
                        t3 = time.perf_counter()
                        self.belief.update_belief(points_stacked, logits_stacked)
                        self.belief.get_belief(scan_points_np)
                        t4 = time.perf_counter()
                        self.times_belief.append(t4 - t0)

                        rendered = self.visualizer.update(
                            scan_points_np,
                            map_points_np,
                            pred_labels_scan,
                            pred_labels_map,
                            self.belief,
                            self.odometry.last_pose,
                        )

                        if not rendered:
                            # pausiert → schweres „Delayed Finalize“ überspringen (optional)
                            pass
                    # Takt einhalten
                    await asyncio.sleep(0.0001)

            except Exception as e:
                print(f"⚠️ consumer error: {e}", file=sys.stderr)
                shutdown_event.set()

        try:
            try:
                prod_task = asyncio.create_task(producer(), name="producer")
                cons_task = asyncio.create_task(consumer(), name="consumer")
                await shutdown_event.wait()
                for t in (prod_task, cons_task):
                    t.cancel()
                # gather, damit CancelledError konsumiert wird
                try:
                    await asyncio.gather(prod_task, cons_task)
                except asyncio.CancelledError:
                    pass
            finally:
                # Client sauber schließen, falls nötig
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        await asyncio.to_thread(close)
                    except Exception as e:
                        print(f"⚠️ client close error: {e}", file=sys.stderr)
        except Exception as e:
            print(f"⚠️ pipeline error: {e}", file=sys.stderr)
            raise


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
