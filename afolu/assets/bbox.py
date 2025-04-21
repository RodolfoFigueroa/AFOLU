from pathlib import Path

import ee
import geopandas as gpd
import rasterio as rio
import rasterio.features as rio_features
import shapely

import dagster as dg
from afolu.resources import PathResource


# Amazonas
@dg.asset(
    name="shapely",
    key_prefix=["amazon", "bbox"],
    io_manager_key="geodataframe_manager",
    group_name="amazon_bbox",
)
def bbox_amazon(path_resource: PathResource) -> gpd.GeoDataFrame:
    fpath = (
        Path(path_resource.data_path) / "initial" / "sdat_671_1_20250409_130228387.tif"
    )
    with rio.open(fpath) as ds:
        data = ds.read(1)
        crs = ds.crs
        transform = ds.transform

    shapes = rio_features.shapes(data, transform=transform)
    polygons = [shapely.geometry.shape(shape) for shape, value in shapes if value == 1]
    polygons = gpd.GeoSeries(polygons, crs=crs).to_crs("ESRI:102033")
    merged = shapely.union_all(polygons.values)

    if isinstance(merged, shapely.MultiPolygon):
        max_area, max_poly = 0, None
        for poly in merged.geoms:
            area = poly.area
            if area > max_area:
                max_area = area
                max_poly = poly
    elif isinstance(merged, shapely.Polygon):
        max_poly = merged
    else:
        err = f"Expected MultiPolygon or Polygon, got {type(merged)}"
        raise TypeError(err)

    simplified = shapely.simplify(max_poly, tolerance=100)

    if not isinstance(simplified, shapely.Polygon):
        err = f"Expected Polygon, got {type(simplified)}"
        raise TypeError(err)

    return gpd.GeoDataFrame(
        geometry=[simplified],
        crs="ESRI:102033",
    ).to_crs("EPSG:4326")


@dg.asset(
    name="shapely",
    key_prefix=["mexico", "bbox"],
    io_manager_key="geodataframe_manager",
    group_name="mexico_bbox",
)
def bbox_mexico(path_resource: PathResource) -> gpd.GeoDataFrame:
    fpath = Path(path_resource.data_path) / "initial" / "gadm41_MEX.gpkg"
    geom: shapely.MultiPolygon = (
        gpd.read_file(fpath, layer=0)["geometry"].to_crs("EPSG:6372").item()
    )

    max_geom, max_area = None, 0
    for g in geom.geoms:
        area = g.area
        if area > max_area:
            max_geom = g
            max_area = area

    simplified = shapely.simplify(max_geom, tolerance=100)

    if not isinstance(simplified, shapely.Polygon):
        err = f"Expected Polygon, got {type(simplified)}"
        raise TypeError(err)

    return gpd.GeoDataFrame(geometry=[simplified], crs="EPSG:6372").to_crs("EPSG:4326")


@dg.asset(
    name="shapely",
    key_prefix=["small", "bbox"],
    io_manager_key="geodataframe_manager",
    group_name="small_bbox",
)
def bbox_small(path_resource: PathResource) -> gpd.GeoDataFrame:
    merged_path = (
        Path(path_resource.population_grids_path)
        / "final"
        / "reprojected"
        / "merged"
        / "19.1.01.gpkg"
    )
    bounds = gpd.read_file(merged_path).to_crs("EPSG:4326")["geometry"].total_bounds
    return gpd.GeoDataFrame(geometry=[shapely.box(*bounds)], crs="EPSG:4326")


def bbox_ee_factory(top_prefix: str) -> dg.AssetsDefinition:
    @dg.asset(
        name="ee",
        key_prefix=[top_prefix, "bbox"],
        ins={"df_bbox": dg.AssetIn([top_prefix, "bbox", "shapely"])},
        io_manager_key="ee_manager",
        group_name=f"{top_prefix}_bbox",
    )
    def _asset(df_bbox: gpd.GeoDataFrame) -> ee.geometry.Geometry:
        bbox_shapely = df_bbox["geometry"].item()

        if not isinstance(bbox_shapely, shapely.Polygon):
            err = f"Expected Polygon, got {type(bbox_shapely)}"
            raise TypeError(err)

        return ee.geometry.Geometry.Polygon(
            list(zip(*bbox_shapely.exterior.coords.xy, strict=False)),
        )

    return _asset


dassets = [bbox_ee_factory(prefix) for prefix in ["amazon", "mexico", "small"]]
