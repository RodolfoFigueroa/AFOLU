from itertools import product

from afolu.assets.constants import LABEL_LIST
from dagster import StaticPartitionsDefinition

year_partitions = StaticPartitionsDefinition([f"{year}" for year in range(2000, 2023)])

year_pair_partitions = StaticPartitionsDefinition(
    [f"{year}_{year + 1}" for year in range(2000, 2022)],
)

label_pair_partitions = StaticPartitionsDefinition(
    [f"{comb[0]}-{comb[1]}" for comb in product(LABEL_LIST, LABEL_LIST)],
)

label_partitions = StaticPartitionsDefinition(LABEL_LIST)
