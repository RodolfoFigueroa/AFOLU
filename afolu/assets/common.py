import ee

import dagster as dg
import pandas as pd


def year_to_band_name(year: int | str) -> str:
    if isinstance(year, str):
        year = int(year)
    return f"b{year - 1999}"


@dg.op
def reduce_image_by_area(img: ee.Image, bbox: ee.Geometry) -> dict:
    transition_img = img.addBands(ee.Image.pixelArea()).select(["area", "class"])
    reduced = transition_img.reduceRegion(
        reducer=(ee.Reducer.sum().group(groupField=1, groupName="transition")),
        scale=30,
        geometry=bbox,
    ).getInfo()

    return reduced


@dg.op(out=dg.Out(io_manager_key="dataframe_manager"))
def reduced_to_table(reduced: dict):
    return (
        pd.DataFrame(reduced["groups"])
        .rename(columns={"sum": "total_area"})
        .assign(transition=lambda df: df.transition.astype(int))
        .set_index("transition")["total_area"]
        .divide(1e6)
        .sort_index()
    )


def area_table_factory(
    name: str, in_name: str, partitions_def: dg.PartitionsDefinition
):
    @dg.graph_asset(
        name=name,
        ins={
            "transition_img": dg.AssetIn(in_name),
            "bbox": dg.AssetIn(["bbox", "ee"]),
        },
        partitions_def=partitions_def,
    )
    def _asset(transition_img: ee.Image, bbox: ee.Geometry):
        return reduced_to_table(reduce_image_by_area(transition_img, bbox))

    return _asset
