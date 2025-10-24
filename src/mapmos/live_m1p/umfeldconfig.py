from typing import Literal, Annotated
from pydantic import Field
from eato_base import NodeConfig

class GenieNodeConfig(NodeConfig):
    kind: Literal["genie"] = "genie"
    resolution: tuple[int, int]
    serial: str
    exposure: float
    gain: float
    image_raw: bool = False

class LidarNodeConfig(NodeConfig):
    kind: Literal["robosense"] = "robosense"
    type: str
    group_address: str
    host_address: str

SensorCfg = Annotated[GenieNodeConfig | LidarNodeConfig, Field(discriminator="kind")]

class UmfeldNodeConfig(NodeConfig):
    sensor_setup: dict[str, SensorCfg]