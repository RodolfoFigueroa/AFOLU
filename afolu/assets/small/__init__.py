import dagster as dg
from afolu.assets.small import areas, transitions

defs = dg.Definitions(
    assets=dg.load_assets_from_modules([areas, transitions]),
)
