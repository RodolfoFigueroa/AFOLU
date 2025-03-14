import ee
import toml

import dagster as dg

import afolu.assets as assets
from afolu.managers import (
    DataFrameManager,
    EarthEngineManager,
    JSONManager,
    NumPyManager,
    ShapelyManager,
)
from afolu.partitions import year_partitions
from afolu.resources import AFOLUClassMapResource, PathResource


ee.Initialize(project="ee-ursa-test")


# Resources
path_resource = PathResource(
    data_path=dg.EnvVar("DATA_PATH"), ghsl_path=dg.EnvVar("GHSL_PATH")
)

with open("./id_map.toml", "r", encoding="utf8") as f:
    config = toml.load(f)

class_map_resource = AFOLUClassMapResource(**config)


# Managers
dataframe_manager = DataFrameManager(path_resource=path_resource)
ee_manager = EarthEngineManager(path_resource=path_resource)
json_manager = JSONManager(path_resource=path_resource)
numpy_manager = NumPyManager(path_resource=path_resource)
shapely_manager = ShapelyManager(path_resource=path_resource)


# Definitions
defs = dg.Definitions(
    assets=(
        dg.load_assets_from_modules([assets.bbox], group_name="bbox")
        + dg.load_assets_from_modules([assets.load], group_name="load")
        + dg.load_assets_from_modules([assets.class_masks], group_name="class_masks")
        + dg.load_assets_from_modules([assets.labels], group_name="labels")
        + dg.load_assets_from_modules([assets.transitions], group_name="transitions")
    ),
    resources=dict(
        class_map_resource=class_map_resource,
        path_resource=path_resource,
        dataframe_manager=dataframe_manager,
        ee_manager=ee_manager,
        json_manager=json_manager,
        numpy_manager=numpy_manager,
        shapely_manager=shapely_manager,
        year_partitions=year_partitions,
    ),
)
