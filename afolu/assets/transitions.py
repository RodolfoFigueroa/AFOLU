import ee


def transition_raster():
    start_map = img_split_map[start_year]
    end_map = img_split_map[end_year]

    masked = ee.Image.constant(0).rename("class").uint8()

    for start_label in LABEL_LIST:
        for end_label in LABEL_LIST:
            start_img = start_map[start_label].rename("class")
            end_img = end_map[end_label].rename("class")

            masked = masked.where(
                start_img.bitwiseAnd(end_img),
                transition_label_map[(start_label, end_label)],
            )

    masked = masked.addBands(ee.Image.pixelArea()).select(["area", "class"])
    reduced = masked.reduceRegion(
        reducer=(ee.Reducer.sum().group(groupField=1, groupName="transition")),
        scale=30,
        geometry=bbox,
    ).getInfo()
