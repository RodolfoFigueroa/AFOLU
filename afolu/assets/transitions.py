import ee

import dagster as dg
import numpy as np
import pandas as pd

from afolu.assets.common import year_to_band_name
from afolu.assets.constants import LABEL_LIST
from afolu.partitions import year_partitions


ins = {f"{label}_img": dg.AssetIn(["class_mask", label]) for label in LABEL_LIST}
ins["grassland_img"] = dg.AssetIn(["class_mask", "grassland_merged"])


@dg.asset(ins=ins, partitions_def=year_partitions, io_manager_key="ee_manager")
def transition_raster(
    context: dg.AssetExecutionContext,
    transition_label_map: dict[str, list[str, str]],
    cropland_img: ee.Image,
    flooded_img: ee.Image,
    forest_primary_img: ee.Image,
    forest_secondary_img: ee.Image,
    grassland_img: ee.Image,
    mangrove_img: ee.Image,
    other_img: ee.Image,
    pasture_img: ee.Image,
    settlement_img: ee.Image,
    shrubland_img: ee.Image,
    wetland_img: ee.Image,
) -> ee.Image:
    start_year, end_year = context.partition_key.split("_")
    start_band = year_to_band_name(start_year)
    end_band = year_to_band_name(end_year)

    masked = ee.Image.constant(0).rename("class").uint8()

    img_map = {
        "cropland": cropland_img,
        "flooded": flooded_img,
        "forest_primary": forest_primary_img,
        "forest_secondary": forest_secondary_img,
        "grassland": grassland_img,
        "mangrove": mangrove_img,
        "other": other_img,
        "pasture": pasture_img,
        "settlement": settlement_img,
        "shrubland": shrubland_img,
        "wetland": wetland_img,
    }

    transition_label_map_inv = {
        tuple(value): int(key) for key, value in transition_label_map.items()
    }

    for start_label, start_img in img_map.items():
        for end_label, end_img in img_map.items():
            a = start_img.select(start_band).rename("class")
            b = end_img.select(end_band).rename("class")

            masked = masked.where(
                a.bitwiseAnd(b),
                transition_label_map_inv[(start_label, end_label)],
            )

    return masked


@dg.op
def reduce_image(img: ee.Image, bbox: ee.Geometry) -> dict:
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


@dg.graph_asset(
    ins={
        "transition_img": dg.AssetIn("transition_raster"),
        "bbox": dg.AssetIn(["bbox", "ee"]),
    },
    partitions_def=year_partitions,
)
def transition_table(transition_img: ee.Image, bbox: ee.Geometry):
    reduced = reduce_image(transition_img, bbox)
    return reduced_to_table(reduced)


# pylint: disable=redefined-outer-name
@dg.asset(partitions_def=year_partitions, io_manager_key="dataframe_manager")
def transition_cross(
    transition_table: pd.DataFrame, transition_label_map: dict[str, list[str, str]]
) -> pd.DataFrame:
    transition_table = transition_table.set_index("transition")["total_area"]

    transition_label_map = pd.Series(
        {int(key): value for key, value in transition_label_map.items()},
        name="transition",
    )

    return (
        pd.concat([transition_table, transition_label_map], axis=1)
        .assign(
            start=lambda df: df.transition.str[0], end=lambda df: df.transition.str[1]
        )
        .drop(columns=["transition"])
        .pivot(index="start", columns="end", values="total_area")
    )


@dg.asset(partitions_def=year_partitions, io_manager_key="dataframe_manager")
def transition_cross_fixed(transition_cross: pd.DataFrame):
    transition_cross = transition_cross.set_index("start")

    for start in LABEL_LIST:
        if start == "forest_primary":
            continue

        transition_cross.loc[start, "forest_secondary"] = np.nansum(
            [
                transition_cross.loc[start, "forest_secondary"],
                transition_cross.loc[start, "forest_primary"],
            ]
        )
        transition_cross.loc[start, "forest_primary"] = np.nan

    transition_cross = transition_cross.fillna(0)
    return transition_cross


@dg.asset(partitions_def=year_partitions, io_manager_key="dataframe_manager")
def transition_cross_frac(transition_cross_fixed: pd.DataFrame) -> pd.DataFrame:
    transition_cross_fixed = transition_cross_fixed.set_index("start")

    zero_rows = transition_cross_fixed.index[transition_cross_fixed.sum(axis=1) == 0]
    for elem in zero_rows:
        transition_cross_fixed.loc[elem, elem] = 1

    return transition_cross_fixed.divide(transition_cross_fixed.sum(axis=1), axis=0)


@dg.asset(io_manager_key="numpy_manager")
def transition_cube(transition_cross_frac: dict[str, pd.DataFrame]) -> np.ndarray:
    out_arr = np.empty(
        (len(transition_cross_frac), len(LABEL_LIST), len(LABEL_LIST)), dtype=float
    )

    for i, key in enumerate(sorted(list(transition_cross_frac.keys()))):
        df = transition_cross_frac[key].set_index("start").sort_index()
        out_arr[i] = df.to_numpy()
    return out_arr
