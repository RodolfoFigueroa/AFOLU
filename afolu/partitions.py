from dagster import StaticPartitionsDefinition


year_partitions = StaticPartitionsDefinition(
    [f"{year}_{year + 1}" for year in range(2000, 2022)]
)
