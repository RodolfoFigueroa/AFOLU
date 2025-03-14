import ee
import ee.deserializer
import json
import shapely

import dagster as dg
import numpy as np
import pandas as pd

from afolu.resources import PathResource
from pathlib import Path


class BaseManager(dg.ConfigurableIOManager):
    extension: str
    path_resource: dg.ResourceDependency[PathResource]

    def _get_path(
        self, context: dg.InputContext | dg.OutputContext
    ) -> Path | dict[str, Path]:
        out_path = Path(self.path_resource.data_path) / "generated"
        fpath = out_path / "/".join(context.asset_key.path)

        if context.has_asset_partitions:
            if len(context.asset_partition_keys) == 1:
                final_path = fpath / context.asset_partition_key
                final_path = final_path.with_suffix(final_path.suffix + self.extension)
            else:
                final_path = {}
                for key in context.asset_partition_keys:
                    temp_path = fpath / key
                    temp_path = temp_path.with_suffix(temp_path.suffix + self.extension)
                    final_path[key] = temp_path
        else:
            final_path = fpath.with_suffix(fpath.suffix + self.extension)
        return final_path


class BaseJSONManager(BaseManager):
    def __init__(self, *args, **kwargs):
        kwargs["extension"] = ".json"
        super().__init__(*args, **kwargs)

    def _write_serialized_json(self, serialized: dict, context: dg.OutputContext):
        fpath = self._get_path(context)
        fpath.parent.mkdir(exist_ok=True, parents=True)

        with open(fpath, "w", encoding="utf8") as f:
            json.dump(serialized, f)

    def _read_serialized_json(self, context: dg.InputContext) -> dict:
        fpath = self._get_path(context)
        with open(fpath, "r", encoding="utf8") as f:
            serialized = json.load(f)
        return serialized


class JSONManager(BaseJSONManager):
    def handle_output(self, context: dg.OutputContext, obj: dict):
        self._write_serialized_json(obj, context)

    def load_input(self, context: dg.InputContext):
        return self._read_serialized_json(context)


class EarthEngineManager(BaseJSONManager):
    def handle_output(self, context: dg.OutputContext, obj: ee.Image | ee.Geometry):
        serialized = json.loads(obj.serialize())
        self._write_serialized_json(serialized, context)

    def load_input(self, context: dg.InputContext) -> ee.Image | ee.Geometry:
        serialized = self._read_serialized_json(context)
        return ee.deserializer.decode(serialized)


class ShapelyManager(BaseJSONManager):
    def handle_output(self, context: dg.OutputContext, obj: shapely.Geometry):
        serialized = json.loads(shapely.to_geojson(obj))
        self._write_serialized_json(serialized, context)

    def load_input(self, context: dg.InputContext) -> shapely.Geometry:
        serialized = self._read_serialized_json(context)
        return shapely.geometry.shape(serialized)


class DataFrameManager(BaseManager):
    def __init__(self, *args, **kwargs):
        kwargs["extension"] = ".csv"
        super().__init__(*args, **kwargs)

    def handle_output(self, context: dg.OutputContext, obj: pd.DataFrame):
        fpath = self._get_path(context)
        fpath.parent.mkdir(exist_ok=True, parents=True)
        obj.to_csv(fpath)

    def load_input(
        self, context: dg.InputContext
    ) -> pd.DataFrame | dict[str, pd.DataFrame]:
        fpath = self._get_path(context)
        if isinstance(fpath, Path):
            return pd.read_csv(fpath)
        else:
            return {key: pd.read_csv(p) for key, p in fpath.items()}


class NumPyManager(BaseManager):
    def __init__(self, *args, **kwargs):
        kwargs["extension"] = ".npy"
        super().__init__(*args, **kwargs)

    def handle_output(self, context: dg.OutputContext, obj: np.ndarray):
        fpath = self._get_path(context)
        np.save(fpath, obj)

    def load_input(self, context: dg.InputContext) -> np.ndarray:
        fpath = self._get_path(context)
        return np.load(fpath)
