import ee
import ee.deserializer
import json
import shapely

import dagster as dg

from afolu.resources import PathResource
from pathlib import Path


class BaseManager(dg.ConfigurableIOManager):
    path_resource: dg.ResourceDependency[PathResource]

    def _get_path(self, context: dg.InputContext | dg.OutputContext) -> Path:
        out_path = Path(self.path_resource.data_path) / "generated"
        fpath = out_path / "/".join(context.asset_key.path)
        return fpath


class BaseJSONManager(BaseManager):
    def _write_serialized_json(self, serialized: dict, context: dg.OutputContext):
        fpath = self._get_path(context)
        fpath = fpath.with_suffix(".json")
        fpath.parent.mkdir(exist_ok=True, parents=True)

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(serialized, f)

    def _load_serialized_json(self, context: dg.InputContext) -> dict:
        fpath = self._get_path(context)
        fpath = fpath.with_suffix(".json")
        with open(fpath, "r", encoding="utf8") as f:
            serialized = json.load(f)
        return serialized


class EarthEngineManager(BaseJSONManager):
    def handle_output(self, context: dg.OutputContext, obj: ee.Image | ee.Geometry):
        serialized = json.loads(obj.serialize())
        self._write_serialized_json(serialized, context)

    def load_input(self, context: dg.InputContext) -> ee.Image | ee.Geometry:
        serialized = self._load_serialized_json(context)
        return ee.deserializer.decode(serialized)


class ShapelyManager(BaseJSONManager):
    def handle_output(self, context: dg.OutputContext, obj: shapely.Geometry):
        serialized = json.loads(shapely.to_geojson(obj))
        self._write_serialized_json(serialized, context)

    def load_input(self, context: dg.InputContext) -> shapely.Geometry:
        serialized = self._load_serialized_json(context)
        return shapely.geometry.shape(serialized)
