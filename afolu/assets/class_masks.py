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
        "forests_img": dg.AssetIn(["class_mask", "forests"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def forests_primary(forests_img: ee.Image, forests_mask: ee.Image):
    return forests_img.bitwiseAnd(forests_mask)


@dg.asset(
    ins={
        "forests_img": dg.AssetIn(["class_mask", "forests"]),
        "forests_primary_img": dg.AssetIn(["class_mask", "forests_primary"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def forests_secondary(forests_img: ee.Image, forests_primary_img: ee.Image):
    return forests_img.bitwiseAnd(forests_primary_img.bitwiseNot())


@dg.asset(
    ins={
        "grasslands_to_pastures_img": dg.AssetIn(
            ["class_mask", "grasslands_to_pastures"]
        ),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def pastures(grasslands_to_pastures_img: ee.Image, pastures_random_mask: ee.Image):
    return grasslands_to_pastures_img.bitwiseAnd(pastures_random_mask)


@dg.asset(
    ins={
        "grasslands_to_pastures_img": dg.AssetIn(
            ["class_mask", "grasslands_to_pastures"]
        ),
        "grasslands_img": dg.AssetIn(["class_mask", "grasslands"]),
    },
    key_prefix="class_mask",
    io_manager_key="ee_manager",
)
def grasslands_merged(
    grasslands_to_pastures_img: ee.Image,
    grasslands_img: ee.Image,
    pastures_random_mask: ee.Image,
):
    return grasslands_to_pastures_img.bitwiseAnd(
        pastures_random_mask.bitwiseNot()
    ).bitwiseOr(grasslands_img)


assets = [
    class_mask_factory(class_name)
    for class_name in (
        "croplands",
        "forests_mangroves",
        "forests",
        "grasslands",
        "grasslands_to_pastures",
        "wetlands",
        "settlements",
        "flooded",
        "shrublands",
        "other",
    )
]
