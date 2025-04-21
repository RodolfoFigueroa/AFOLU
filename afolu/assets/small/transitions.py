import ee
import numpy as np
import pandas as pd

import dagster as dg
from afolu.assets.common import (
    transition_table_fixed_factory,
    transition_table_frac_factory,
    transition_cube_factory,
    year_to_band_name,
)
from afolu.assets.constants import LABEL_LIST
from afolu.partitions import year_pair_partitions


@dg.asset(io_manager_key="json_manager", group_name="small_transition")
def transition_label_map() -> dict[int, list[str]]:
    multiplier = 10 ** np.ceil(np.log10(len(LABEL_LIST)))

    out = {}
    for i, start_label in enumerate(LABEL_LIST):
        for j, end_label in enumerate(LABEL_LIST):
            key = int(i * multiplier + j)
            if key in out:
                err = f"Key {key} already exists in the dictionary."
                raise ValueError(err)
            out[key] = [start_label, end_label]

    return out


ins = {
    f"{label}_img": dg.AssetIn(["small", "class_mask", label]) for label in LABEL_LIST
}
ins["grasslands_img"] = dg.AssetIn(["small", "class_mask", "grasslands_merged"])
ins["bbox"] = dg.AssetIn(["small", "bbox", "ee"])


@dg.asset(
    ins=ins,
    name="raster",
    key_prefix=["small", "transition"],
    partitions_def=year_pair_partitions,
    io_manager_key="ee_manager",
    group_name="small_transition",
)
def transition_raster(
    context: dg.AssetExecutionContext,
    bbox: ee.geometry.Geometry,
    transition_label_map: dict[str, list[str]],
    croplands_img: ee.image.Image,
    flooded_img: ee.image.Image,
    forests_mangroves_img: ee.image.Image,
    forests_primary_img: ee.image.Image,
    forests_secondary_img: ee.image.Image,
    grasslands_img: ee.image.Image,
    other_img: ee.image.Image,
    pastures_img: ee.image.Image,
    settlements_img: ee.image.Image,
    shrublands_img: ee.image.Image,
    wetlands_img: ee.image.Image,
) -> ee.image.Image:
    start_year, end_year = context.partition_key.split("_")
    start_band = year_to_band_name(start_year)
    end_band = year_to_band_name(end_year)

    masked = ee.image.Image.constant(0).rename("class").uint8().clip(bbox)

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

    transition_label_map_inv = {
        tuple(value): int(key) for key, value in transition_label_map.items()
    }

    for start_label, start_img in img_map.items():
        for end_label, end_img in img_map.items():
            a: ee.image.Image = start_img.select(start_band).rename("class")
            b: ee.image.Image = end_img.select(end_band).rename("class")

            masked = masked.where(
                a.And(b),
                transition_label_map_inv[(start_label, end_label)],
            )

    return masked


@dg.asset(
    name="table",
    key_prefix=["small", "transition"],
    ins={
        "raster": dg.AssetIn(["small", "transition", "raster"]),
        "bbox": dg.AssetIn(["small", "bbox", "ee"]),
    },
    partitions_def=year_pair_partitions,
    io_manager_key="dataframe_manager",
    group_name="small_transition",
)
def transition_table(
    raster: ee.image.Image,
    bbox: ee.geometry.Geometry,
    transition_label_map: dict[str, list[str]],
) -> pd.DataFrame:
    transition_img: ee.image.Image = raster.addBands(ee.image.Image.pixelArea()).select(
        ["area", "class"]
    )

    response = transition_img.reduceRegion(
        reducer=(ee.reducer.Reducer.sum().group(groupField=1, groupName="transition")),
        scale=30,
        geometry=bbox,
        maxPixels=int(1e10),
    ).getInfo()

    if response is None:
        err = "No data returned from reduceRegion."
        raise ValueError(err)

    rows = []
    for elem in response["groups"]:
        rows.append(
            {  # noqa: PERF401
                "label": transition_label_map[str(elem["transition"])],
                "area": float(elem["sum"]),
            }
        )

    out = (
        pd.DataFrame(rows)
        .assign(
            start=lambda df: df["label"].str[0],
            end=lambda df: df["label"].str[1],
        )
        .drop(columns=["label"])
        .pivot_table(index="start", columns="end", values="area")
        .fillna(0)
    )

    for label in LABEL_LIST:
        if label not in out.index:
            out.loc[label] = 0

        if label not in out.columns:
            out[label] = 0

    out = out.sort_index()
    out = out[sorted(out.columns)]

    if not isinstance(out, pd.DataFrame):
        err = f"Expected pd.DataFrame, got {type(out)}"
        raise TypeError(err)

    return out


dassets = [
    transition_table_fixed_factory("small"),
    transition_table_frac_factory("small"),
    transition_cube_factory("small"),
]
