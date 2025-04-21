import ee

import dagster as dg


def glc30_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="glc30",
        key_prefix=top_prefix,
        ins={"bbox": dg.AssetIn([top_prefix, "bbox", "ee"])},
        io_manager_key="ee_manager",
        group_name=f"{top_prefix}_load",
    )
    def _asset(bbox: ee.geometry.Geometry) -> ee.image.Image:
        return (
            ee.imagecollection.ImageCollection(
                "projects/sat-io/open-datasets/GLC-FCS30D/annual"
            )
            .filterBounds(bbox)
            .mode()
            .clip(bbox)
        )

    return _asset


def forests_mask_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="forests_mask",
        key_prefix=top_prefix,
        ins={"bbox": dg.AssetIn([top_prefix, "bbox", "ee"])},
        io_manager_key="ee_manager",
        group_name=f"{top_prefix}_load",
    )
    def _asset(bbox: ee.geometry.Geometry) -> ee.image.Image:
        return (
            ee.imagecollection.ImageCollection(
                "NASA/ORNL/global_forest_classification_2020/V1"
            )
            .filterBounds(bbox)
            .mode()
            .clip(bbox)
            .unmask(0)
            .eq(1)
        )

    return _asset


def pastures_random_mask_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="pastures_random_mask",
        key_prefix=top_prefix,
        ins={
            "bbox": dg.AssetIn([top_prefix, "bbox", "ee"]),
            "glc30": dg.AssetIn([top_prefix, "glc30"]),
        },
        io_manager_key="ee_manager",
        group_name=f"{top_prefix}_load",
    )
    def _asset(bbox: ee.geometry.Geometry, glc30: ee.image.Image) -> ee.image.Image:
        proj = glc30.projection().getInfo()

        if not isinstance(proj, dict):
            err = f"Expected dict, got {type(proj)}"
            raise TypeError(err)

        return (
            ee.image.Image.random(42)
            .reproject(crs=proj["crs"], crsTransform=proj["transform"])
            .lte(0.4)
            .clip(bbox)
        )

    return _asset


dassets = [
    factory(top_prefix)
    for factory in [glc30_factory, forests_mask_factory, pastures_random_mask_factory]
    for top_prefix in ["amazon", "mexico", "small"]
]
