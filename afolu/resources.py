import dagster as dg


class PathResource(dg.ConfigurableResource):
    ghsl_path: str
    data_path: str


class AFOLUClassMapResource(dg.ConfigurableResource):
    cropland: list[int]
    forest: list[int]
    grassland: list[int]
    grassland_to_pasture: list[int]
    wetland: list[int]
    mangrove: list[int]
    settlement: list[int]
    flooded: list[int]
    shrubland: list[int]
    other: list[int]
