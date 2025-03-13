import ee
import shapely

import dagster as dg


@dg.asset(name="shapely", key_prefix="bbox", io_manager_key="shapely_manager")
def bbox_shapely() -> shapely.Polygon:
    xmin = -75.73013440795755
    ymin = 1.5033306279740193
    xmax = -75.48956557487034
    ymax = 1.7319518677824375
    return shapely.box(xmin, ymin, xmax, ymax)


# pylint: disable=redefined-outer-name
@dg.asset(
    name="ee",
    key_prefix="bbox",
    ins={"bbox_shapely": dg.AssetIn(["bbox", "shapely"])},
    io_manager_key="ee_manager",
)
def bbox_ee(bbox_shapely: shapely.Polygon) -> ee.Geometry:
    return ee.Geometry.Polygon(list(zip(*bbox_shapely.exterior.coords.xy)))
