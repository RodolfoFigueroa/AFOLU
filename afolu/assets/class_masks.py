import ee

import dagster as dg

from afolu.resources import AFOLUClassMapResource


def class_mask_factory(class_name: str) -> dg.AssetsDefinition:
    @dg.asset(name=class_name, key_prefix="class_mask", io_manager_key="ee_manager")
    def _asset(class_map_resource: AFOLUClassMapResource, glc30: ee.Image):
        label_list = getattr(class_map_resource, class_name)
        mask = glc30.eq(label_list[0])
        for label in label_list[1:]:
            mask = mask.bitwiseOr(glc30.eq(label))
        return mask

    return _asset


@dg.asset(
    ins={
        "forest_img": dg.AssetIn(["class_mask", "forest"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def forest_primary(forest_img: ee.Image, forest_mask: ee.Image):
    return forest_img.bitwiseAnd(forest_mask)


@dg.asset(
    ins={
        "forest_img": dg.AssetIn(["class_mask", "forest"]),
        "forest_primary_img": dg.AssetIn(["class_mask", "forest_primary"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def forest_secondary(forest_img: ee.Image, forest_primary_img: ee.Image):
    return forest_img.bitwiseAnd(forest_primary_img.bitwiseNot())


@dg.asset(
    ins={
        "grassland_to_pasture_img": dg.AssetIn(["class_mask", "grassland_to_pasture"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def pasture(grassland_to_pasture_img: ee.Image, pasture_random_mask: ee.Image):
    return grassland_to_pasture_img.bitwiseAnd(pasture_random_mask)


@dg.asset(
    ins={
        "grassland_to_pasture_img": dg.AssetIn(["class_mask", "grassland_to_pasture"]),
        "grassland_img": dg.AssetIn(["class_mask", "grassland"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def grassland_merged(
    grassland_to_pasture_img: ee.Image,
    grassland_img: ee.Image,
    pasture_random_mask: ee.Image,
):
    return grassland_to_pasture_img.bitwiseAnd(
        pasture_random_mask.bitwiseNot()
    ).bitwiseOr(grassland_img)


assets = [
    class_mask_factory(class_name)
    for class_name in (
        "cropland",
        "forest",
        "grassland",
        "grassland_to_pasture",
        "wetland",
        "mangrove",
        "settlement",
        "flooded",
        "shrubland",
        "other",
    )
]
