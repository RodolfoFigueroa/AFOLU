import ee
import toml

import dagster as dg
from afolu import assets
from afolu.managers import (
    DataFrameManager,
    EarthEngineManager,
    GeoDataFrameManager,
    JSONManager,
    NumPyManager,
    RasterManager,
    ShapelyManager,
    TextManager,
)
from afolu.partitions import (
    label_pair_partitions,
    year_pair_partitions,
    year_partitions,
)
from afolu.resources import (
    AFOLUClassMapResource,
    LabelResource,
    PathResource,
    SelectedAreaResource,
)

ee.Initialize(project="ee-ursa-test")


# Resources
selected_area_resource = SelectedAreaResource(selected_area="mexico")

path_resource = PathResource(
    data_path=dg.EnvVar("DATA_PATH"),
    ghsl_path=dg.EnvVar("GHSL_PATH"),
    population_grids_path=dg.EnvVar("POPULATION_GRIDS_PATH"),
)

with open("./id_map.toml", encoding="utf8") as f:
    config = toml.load(f)

spec_map = {}
all_resources_map = {}
for key, spec in config.items():

    resource = LabelResource(
        ranges=spec.get("ranges"),
    )

    spec_map[key] = resource
    all_resources_map[f"{key}_map"] = resource

class_map_resource = AFOLUClassMapResource(**spec_map)

# Managers
dataframe_manager = DataFrameManager(
    path_resource=path_resource,
    extension=".csv",
)
ee_manager = EarthEngineManager(
    path_resource=path_resource,
    extension=".json",
)
geodataframe_manager = GeoDataFrameManager(
    path_resource=path_resource,
    extension=".gpkg",
)
json_manager = JSONManager(
    path_resource=path_resource,
    extension=".json",
)
numpy_manager = NumPyManager(
    path_resource=path_resource,
    extension=".npy",
)
raster_manager = RasterManager(
    path_resource=path_resource,
    extension=".tif",
)
shapely_manager = ShapelyManager(
    path_resource=path_resource,
    extension=".json",
)
text_manager = TextManager(
    path_resource=path_resource,
    extension=".txt",
)


# Definitions
defs = dg.Definitions.merge(
    dg.Definitions(
        assets=(
            list(
                dg.load_assets_from_modules(
                    [
                        assets.bbox,
                        assets.class_masks,
                        assets.load,
                    ],
                )
            )
        ),
        resources=dict(
            class_map_resource=class_map_resource,
            path_resource=path_resource,
            dataframe_manager=dataframe_manager,
            ee_manager=ee_manager,
            geodataframe_manager=geodataframe_manager,
            json_manager=json_manager,
            numpy_manager=numpy_manager,
            raster_manager=raster_manager,
            shapely_manager=shapely_manager,
            year_partitions=year_partitions,
            year_pair_partitions=year_pair_partitions,
            label_pair_partitions=label_pair_partitions,
            text_manager=text_manager,
            selected_area_resource=selected_area_resource,
            **all_resources_map,
        ),
    ),
    assets.large.defs,
    assets.small.defs,
)
