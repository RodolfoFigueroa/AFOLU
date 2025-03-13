from afolu.resources import AFOLUClassMapResource


def generate_label_list(class_map_resource: AFOLUClassMapResource) -> list[str]:
    label_list = list(class_map_resource.model_dump().keys())
    label_list.remove("forest")
    label_list.remove("grassland_130")
    label_list.remove("grassland_140")
    label_list.extend(["forest_primary", "forest_secondary", "pastures", "grassland"])
    label_list = sorted(label_list)
    return label_list


def year_to_band_name(year: int) -> str:
    return f"b{year - 1999}"
