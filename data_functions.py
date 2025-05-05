import warnings
from pathlib import Path

import pandas as pd
import sisepuede.core.support_classes as sc
import sisepuede.models.afolu as mafl
from sisepuede.utilities._toolbox import islistlike, merge_output_df_list


##  build input df for areas frac
def build_inputs_to_overwrite(
    dict_data: dict[str, pd.DataFrame],
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
) -> pd.DataFrame:
    """Clean and combine the DataFrames to build inputs that overwrite
    example data.
    """

    ##  GET COMPONENT DFs

    df_areas_inital = clean_areas_frac(
        dict_data["areas_frac"],
        model_afolu,
        time_periods,
    )

    df_area_total = clean_area_total(
        dict_data["areas"],
        model_afolu,
        time_periods,
    )

    df_transitions = dict_data["transitions"]

    ##  COMBINE

    df_out = merge_output_df_list(
        [df_areas_inital, df_area_total, df_transitions],
        model_afolu.model_attributes,
    ).ffill()

    df_out[time_periods.field_time_period] = df_out[
        time_periods.field_time_period
    ].astype(int)

    return df_out


def clean_areas_frac(
    df_in: pd.DataFrame,
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
    field_cat: str = "label",
) -> pd.DataFrame:
    """Fix the input data for initial areas"""

    matt = model_afolu.model_attributes
    modvar = matt.get_variable(model_afolu.modvar_lndu_initial_frac)

    if modvar is None:
        err = "Unable to find variable modvar_lndu_initial_frac"
        raise ValueError(err)

    dict_repl = matt.get_category_replacement_field_dict(modvar)

    # map to data frame
    df_out = (
        df_in.copy()
        .set_index([field_cat])
        .transpose()
        .reset_index(names=time_periods.field_time_period)
        .rename_axis(None, axis=1)
        .rename(columns=dict_repl)
    )

    return modvar.get_from_dataframe(
        df_out,
        expand_to_all_categories=True,
        extraction_logic="any_fill",
        fields_additional=[time_periods.field_time_period],
        fill_value=0.0,
    )


##  build input df for areas frac
def clean_area_total(
    df_in: pd.DataFrame,
    model_afolu: mafl.AFOLU,
    time_periods: sc.TimePeriods,
    field_cat: str = "label",
    units_input: str = "km2",
) -> pd.DataFrame:
    """Fix the input data for initial areas"""

    modvar = model_afolu.model_attributes.get_variable(
        model_afolu.model_socioeconomic.modvar_gnrl_area,
    )

    # some field info
    all_cats = df_in[field_cat].unique()
    field = modvar.fields[0]

    # map to data frame
    df_out = (
        df_in.copy()
        .set_index([field_cat])
        .transpose()
        .reset_index(names=time_periods.field_time_period)
        .rename_axis(None, axis=1)
    )

    df_out[field] = df_out[all_cats].sum(axis=1)

    df_out = modvar.get_from_dataframe(
        df_out,
        expand_to_all_categories=True,
        extraction_logic="any_fill",
        fields_additional=[time_periods.field_time_period],
        fill_value=0.0,
    )

    # finally, rescale
    return model_afolu.model_attributes.rescale_fields_to_target(
        df_out,
        [field],
        modvar,
        {
            "area": (units_input, 1),
        },
    )


def get_full_variable(
    construct: "SISEPUEDEDataConstructs",
    modvar_name: str,
    regions: sc.Regions,
    regions_keep: str | list[str] | None = None,
) -> pd.DataFrame:
    """Read in the historical and projected value for the variable"""

    regions_keep = (
        [regions_keep] if isinstance(regions_keep, (str, int)) else regions_keep
    )
    regions_keep = (
        [regions.return_region_or_iso(x, return_type="iso") for x in regions_keep]
        if islistlike(regions_keep)
        else None
    )

    sub = islistlike(regions_keep)
    sub &= (len(regions_keep) > 0) if sub else False
    dict_subset = None if not sub else {regions.field_iso: regions_keep}

    return construct.read_and_combine_from_output_database(
        modvar_name,
        bound_types="nominal",
        table_types=None,
        dict_subset=dict_subset,
    ).drop(
        columns=[construct.table_types.key, construct.variable_bounds_return_type.key],
    )


def get_vars_from_dbdir(
    path_data: Path,
    modvars: list[str],
) -> pd.DataFrame | None:
    """Read in the historical and projected values for database-based inputs
    from a directory that stores table by model variable.
    """

    df_out = None
    if not path_data.is_dir():
        return None

    for modvar in modvars:
        try:
            path_read = path_data.joinpath(f"{modvar}.csv")
            df_cur = pd.read_csv(path_read)

        except Exception as e:  # noqa: BLE001
            msg = f"Unable to retrieve variable {modvar} from directory: {e}"
            warnings.warn(msg, stacklevel=1)
            continue

        df_out = df_cur if df_out is None else df_out.merge(df_cur, how="inner")

    return df_out
