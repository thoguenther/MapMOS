# src/mapmos/live_m1p/mapmos_node.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
from eato_base import EATONode, EATOTopics, ConfigLoader


# Lazy imports, wie in cli.py
from mapmos.datasets import dataset_factory
from mapmos.pipeline_live import MapMOSPipeline as Pipeline


def _env_path(name: str) -> Optional[Path]:
    v = os.environ.get(name)
    return Path(v) if v else None


def _env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    v = v.strip().lower()
    return v in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    try:
        return int(v)
    except ValueError:
        return default


class MapMOSNode(EATONode):
    """
    Minimaler Node-Wrapper für EATO/eato-base:
    - setup(): optionaler Hook (hier no-op)
    - run(): startet die MapMOS-Pipeline synchron (blockierend)
    """

    def __init__(self):
        super().__init__()
        # Pflichtparameter
        self.weights: Path = _env_path("MAPMOS_WEIGHTS") or Path(
            _env_str("MAPMOS_WEIGHTS", "")
        )
        if not self.weights or not self.weights.exists():
            raise RuntimeError(
                "MAPMOS_WEIGHTS ist nicht gesetzt oder zeigt auf eine nicht existente Datei (*.ckpt)"
            )

        # Datenwurzel (bei Live-Datasets oft egal; Pipeline-API verlangt einen Pfad)
        self.data_dir: Path = _env_path("MAPMOS_DATA_DIR") or Path(
            _env_str("MAPMOS_DATA_DIR", ".")
        )

        # Optional/Dataset-spezifisch
        self.dataloader: Optional[str] = _env_str("MAPMOS_DATALOADER", None)
        self.sequence: Optional[str] = _env_str("MAPMOS_SEQUENCE", None)
        self.topic: Optional[str] = _env_str("MAPMOS_TOPIC", None)
        self.meta: Optional[Path] = _env_path("MAPMOS_META")

        # Pipeline/Runtime
        self.config: Optional[Path] = _env_path("MAPMOS_CONFIG")
        self.visualize: bool = _env_bool("MAPMOS_VISUALIZE", False)
        self.save_ply: bool = _env_bool("MAPMOS_SAVE_PLY", False)
        self.save_kitti: bool = _env_bool("MAPMOS_SAVE_KITTI", False)
        self.n_scans: int = _env_int("MAPMOS_N_SCANS", -1)
        self.jump: int = _env_int("MAPMOS_JUMP", 0)

    # EATO-Hook (falls genutzt)
    def setup(self):
        pass

    # EATO-Hook: wird von deinem System aufgerufen
    def run(self):
        ds = dataset_factory(
            dataloader=self.dataloader,
            data_dir=self.data_dir,
            sequence=self.sequence,
            topic=self.topic,
            meta=self.meta,
        )

        Pipeline(
            dataset=ds,
            weights=self.weights,
            config=self.config,
            visualize=self.visualize,
            save_ply=self.save_ply,
            save_kitti=self.save_kitti,
            n_scans=self.n_scans,
            jump=self.jump,
        ).run().print()

    def run_loop(self):
        # required by EATONode interface – hier z. B. blockierend starten
        self.run()


def mapmosnode() -> MapMOSNode:
    """
    Fabrikfunktion, damit du in deiner cli.py einfach mapmosnode() in die EATOSystem-
    Definition stecken kannst (gleiches Pattern wie bei deinem LidarNode).
    """
    return MapMOSNode()
