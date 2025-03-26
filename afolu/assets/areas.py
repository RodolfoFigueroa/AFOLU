import ee

import dagster as dg
import pandas as pd

from afolu.assets.common import area_table_factory, year_to_band_name
from afolu.assets.constants import LABEL_LIST
from afolu.partitions import year_partitions


ins = {f"{label}_img": dg.AssetIn(["class_mask", label]) for label in LABEL_LIST}
ins["grasslands_img"] = dg.AssetIn(["class_mask", "grasslands_merged"])


@dg.asset(ins=ins, partitions_def=year_partitions, io_manager_key="ee_manager")
def area_raster(
    context: dg.AssetExecutionContext,
    croplands_img: ee.Image,
    flooded_img: ee.Image,
    forests_mangroves_img: ee.Image,
    forests_primary_img: ee.Image,
    forests_secondary_img: ee.Image,
    grasslands_img: ee.Image,
    other_img: ee.Image,
    pastures_img: ee.Image,
    settlements_img: ee.Image,
    shrublands_img: ee.Image,
    wetlands_img: ee.Image,
) -> ee.Image:
    band = year_to_band_name(context.partition_key)

    img_map = {
        "croplands": croplands_img,
        "flooded": flooded_img,
        "forests_mangroves": forests_mangroves_img,
        "forests_primary": forests_primary_img,
        "forests_secondary": forests_secondary_img,
        "grasslands": grasslands_img,
        "other": other_img,
        "pastures": pastures_img,
        "settlements": settlements_img,
        "shrublands": shrublands_img,
        "wetlands": wetlands_img,
    }

    out_img = ee.Image.constant(0).rename("class").uint8()
    for i, label in enumerate(LABEL_LIST):
        out_img = out_img.where(img_map[label].select(band).rename("class"), i + 1)

    return out_img


@dg.asset(
    ins={"area_table_map": dg.AssetIn("area_table")}, io_manager_key="dataframe_manager"
)
def area_table_merged(area_table_map: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = []

    label_map = {i + 1: LABEL_LIST[i] for i in range(len(LABEL_LIST))}
    label_map[0] = "null"

    for i, key in enumerate(sorted(list(area_table_map.keys()))):
        series = (
            area_table_map[key]
            .rename(columns={"transition": "label"})
            .set_index("label")["total_area"]
            .rename(i)
        )
        out.append(series)

    return (
        pd.concat(out, axis=1)
        .reset_index()
        .assign(label=lambda df: df["label"].map(label_map))
        .set_index("label")
        .drop(index=["null"])
    )


@dg.asset(io_manager_key="dataframe_manager")
def area_table_merged_frac(area_table_merged: pd.DataFrame) -> pd.DataFrame:
    df = area_table_merged.set_index("label")
    return df.divide(df.sum(axis=0))


area_table = area_table_factory(
    name="area_table", in_name="area_raster", partitions_def=year_partitions
)
