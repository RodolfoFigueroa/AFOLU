def year_to_band_name(year: int | str) -> str:
    if isinstance(year, str):
        year = int(year)
    return f"b{year - 1999}"
