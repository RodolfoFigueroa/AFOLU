import dagster as dg


class PathResource(dg.ConfigurableResource):
    ghsl_path: str
    data_path: str


class AFOLUClassMapResource(dg.ConfigurableResource):
    croplands: list[int]
    forests: list[int]
    grasslands: list[int]
    grasslands_to_pastures: list[int]
    wetlands: list[int]
    forests_mangroves: list[int]
    settlements: list[int]
    flooded: list[int]
    shrublands: list[int]
    other: list[int]
