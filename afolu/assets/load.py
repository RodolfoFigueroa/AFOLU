import ee

import dagster as dg


@dg.asset(ins={"bbox": dg.AssetIn(["bbox", "ee"])}, io_manager_key="ee_manager")
def glc30(bbox: ee.Geometry) -> ee.Image:
    col = ee.ImageCollection(
        "projects/sat-io/open-datasets/GLC-FCS30D/annual"
    ).filterBounds(bbox)

    assert col.size().getInfo() == 1

    img = col.first().clip(bbox)
    return img


@dg.asset(ins={"bbox": dg.AssetIn(["bbox", "ee"])}, io_manager_key="ee_manager")
def forests_mask(bbox: ee.Geometry) -> ee.Image:
    return (
        ee.ImageCollection("NASA/ORNL/global_forest_classification_2020/V1")
        .filterBounds(bbox)
        .mode()
        .clip(bbox)
        .unmask(0)
        .eq(1)
    )


# pylint: disable=redefined-outer-name
@dg.asset(ins={"bbox": dg.AssetIn(["bbox", "ee"])}, io_manager_key="ee_manager")
def pastures_random_mask(bbox: ee.Geometry, glc30: ee.Image) -> ee.Image:
    proj = glc30.projection().getInfo()
    return (
        ee.Image.random(42)
        .reproject(crs=proj["crs"], crsTransform=proj["transform"])
        .lte(0.4)
        .clip(bbox)
    )
