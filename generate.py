import os
import pathlib
import warnings

import pandas as pd
import sisepuede.core.support_classes as sc
import sisepuede.manager.sisepuede_examples as sxl
import sisepuede.manager.sisepuede_file_structure as sfs
import sisepuede.models.afolu as mafl
import sisepuede.utilities._toolbox as sf

import data_functions as dfuncs

warnings.filterwarnings("ignore")


class InvalidDirectoryError(Exception):
    pass


##########################
#    GLOBAL VARIABLES    #
##########################

# model variables to access
_MODVAR_NAME_EF_CONVERSION = ":math:\\text{CO}_2 Land Use Conversion Emission Factor"
_MODVAR_NAME_SF_FOREST = "Forest Sequestration Emission Factor"
_MODVAR_NAME_SF_FOREST_YOUNG = "Young Secondary Forest Sequestration Emission Factor"
_MODVAR_NAME_SF_LAND_USE = "Land Use Biomass Sequestration Factor"
_MODVARS_ALL_FROM_DB = [
    _MODVAR_NAME_EF_CONVERSION,
    _MODVAR_NAME_SF_FOREST,
    _MODVAR_NAME_SF_FOREST_YOUNG,
    _MODVAR_NAME_SF_LAND_USE,
]

# paths
_PATH_CUR = pathlib.Path(__file__).parents[0]
_PATH_DATA_FROM_URSA_PIPELINE = _PATH_CUR.joinpath("data")
_PATH_DATA_FROM_SSP_PIPELINE = _PATH_CUR.joinpath("sisepuede_pipeline_data")


##############################
#    SUPPORTING FUNCTIONS    #
##############################


def build_dataset(
    examples: sxl.SISEPUEDEExamples,
    region: str | int,
    model_afolu: mafl.AFOLU,
    regions: sc.Regions,
    time_periods: sc.TimePeriods,
    path_data_ssp: os.PathLike | None = None,
    path_data_ursa: os.PathLike | None = None,
) -> pd.DataFrame:
    """Build the dataset for the model runs

    Function Arguments
    ------------------
    examples : SISEPUEDEExamples
        Set of example data for SISEPUEDE. Used as basis for model inputs, as it
        includes reasonable variable values for unused model components (e.g.,
        energy).
    region : Union[str, int]
        Region to use from SISEPUEDE pipeline data. Can be:
            * A region name (see regions.attributes.key_values for valid names)
            * A region ISO-3 Alphanumeric code (regions.all_isos)
            * A region ISO-3 Numeric code (regions.all_isos_numeric)
    model_afolu : AFOLU
        SISEPUEDE AFOLU model
    regions : Regions
        SISEPUEDE Regions object used to manage regions
    time_periods : TimePeriods
        SISEPUEDE TimePeriods object

    Keyword Arguments
    -----------------
    path_data_ssp : Union[str, pathlib.Path, None]
        Optional specification of path to directory containing data from
        SISEPUEDE pipeline
    path_data_ursa : Union[str, pathlib.Path, None]
        Optional specification of path to directory containing data from
        URSA pipeline
    """

    # initialize a base data frame
    df_base = examples("input_data_frame")

    if not isinstance(df_base, pd.DataFrame):
        err = f"Expected pd.DataFrame, got {type(df_base)}"
        raise TypeError(err)

    df_base = df_base.copy()

    # get ursa pipeline outputs
    dict_ursa_data = get_dict_ursa_pipeline_data(path_data_ursa)

    # get from CSV and from DB
    df_inputs_ursa = dfuncs.build_inputs_to_overwrite(
        dict_ursa_data,
        model_afolu,
        time_periods,
    )

    # get files from pipeline
    path_ssp_data = get_path(path_data_ssp, _PATH_DATA_FROM_SSP_PIPELINE)
    df_inputs_ssp = dfuncs.get_vars_from_dbdir(
        path_ssp_data,
        _MODVARS_ALL_FROM_DB,
    )

    if not isinstance(df_inputs_ssp, pd.DataFrame):
        err = f"Expected pd.DataFrame, got {type(df_inputs_ssp)}"
        raise TypeError(err)

    # get region and filter
    df_inputs_ssp = regions.extract_from_df(
        df_inputs_ssp,
        region,
        regions.field_iso,
    )

    if not isinstance(df_inputs_ssp, pd.DataFrame):
        err = "No data found for region '{region}' in SSP data."
        raise TypeError(err)

    # adjust time periods for now
    field_year = time_periods.field_year
    df_inputs_ssp[field_year] -= df_inputs_ssp[field_year].iloc[0]
    df_inputs_ssp = df_inputs_ssp.rename(
        columns={field_year: time_periods.field_time_period},
    )

    ##  SET AGGREGATE INPUTS AND RUN MODEL

    df_in = sf.match_df_to_target_df(
        df_base.drop(columns=[regions.key]),
        df_inputs_ursa,
        [time_periods.field_time_period],
        overwrite_only=False,
        try_interpolate=True,
    )

    return sf.match_df_to_target_df(
        df_in,
        df_inputs_ssp.drop(columns=[regions.field_iso]),
        [time_periods.field_time_period],
        overwrite_only=False,
        try_interpolate=True,
    )


def get_dict_ursa_pipeline_data(
    path: os.PathLike | None = None,
) -> dict[str, pd.DataFrame]:
    """Using a directory that contains the three outputs from IDB Cities
    pipeline (areas_frac.csv, area.csv, transitions.csv), load a dictionary
    of available data.
    """

    # get path
    path_data = get_path(path, _PATH_DATA_FROM_URSA_PIPELINE)

    # read in data of interest
    dict_data = {}
    for p in path_data.iterdir():
        if p.suffix == ".csv":
            nm = p.parts[-1].replace(".csv", "")
            df = pd.read_csv(p)

            dict_data.update({nm: df})

    return dict_data


def get_path(
    path: os.PathLike | None,
    path_default: os.PathLike,
    *,
    verify_exists: bool = True,
) -> pathlib.Path:
    """Check an input path and return a default if needed."""
    # check type of default
    if not isinstance(path_default, pathlib.Path):
        tp = str(type(path_default))
        err = f"Invalid type '{tp}': path must be of type pathlib.Path."
        raise TypeError(err)

    # try formatting input path
    path = pathlib.Path(path) if isinstance(path, str) else path
    path = path_default if not isinstance(path, pathlib.Path) else path

    # verify existence
    if verify_exists and not path.is_dir():
        err = f"Directory '{path}' does not exist. Unable to read input files."
        raise InvalidDirectoryError(err)

    return path


#######################
#    MAIN FUNCTION    #
#######################


def main(
    region: str | int,
    examples: sxl.SISEPUEDEExamples | None = None,
    file_struct: sfs.SISEPUEDEFileStructure | None = None,
    model_afolu: mafl.AFOLU | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the input data and run the model for a given set of URSA inputs in
        a region.

    Function Arguments
    ------------------
    region : Union[str, int]
        Region to use from SISEPUEDE pipeline data for factor access. Can be:
            * A region name
                See `regions.attributes.key_values` for valid names
            * A region ISO-3 Alphanumeric code
                See `regions.all_isos` for valid ISO alphanumeric-3 codes
            * A region ISO-3 Numeric code
                See `regions.all_isos_numeric` for valid ISO numeric codes


    Keyword Arguments
    -----------------
    examples : Union[sxl.SISEPUEDEExamples, None]
        Optional set of example data for SISEPUEDE. Used as basis for model
        inputs, as it includes reasonable variable values for unused model
        components (e.g., energy). If None, main() will create one.

        If iterating, it is more efficient to create one outside of the
        iteration and pass it to each iterand instead of instantiating one at
        each iteration.
    file_struct : Union[sfs.SISEPUEDEFileStructure, None]
        Optional SISEPUEDEFileStructure to pass for access to model attributes,
        regions, etc. If None, then main() will create one.

        If iterating, it is more efficient to create one outside of the
        iteration and pass it to each iterand instead of instantiating one at
        each iteration.
    model_afolu : Union[AFOLU, None]
        Optional AFOLU model to use for method and variable access and
        preservation of internal consistency of data. If None, will create one.

        If iterating, it is more efficient to create one outside of the
        iteration and pass it to each iterand instead of instantiating one at
        each iteration.
    **kwargs:
        Can include


    """
    ##  INITIALIZE SISEPUEDE OBJECTS

    examples = (
        sxl.SISEPUEDEExamples() if not sxl.is_sisepuede_examples(examples) else examples
    )

    if examples is None:
        err = "Examples is None. Unable to access model attributes."
        raise ValueError(err)

    file_struct = (
        sfs.SISEPUEDEFileStructure()
        if not sfs.is_sisepuede_file_structure(file_struct)
        else file_struct
    )

    if file_struct is None:
        err = "File structure is None. Unable to access model attributes."
        raise ValueError(err)

    # set the model attributes
    matt = file_struct.model_attributes

    if matt is None:
        err = "Model attributes are None. Unable to access model attributes."
        raise ValueError(err)

    # objects that depend on model attributes
    model_afolu = mafl.AFOLU(matt)
    regions = sc.Regions(matt)
    time_periods = sc.TimePeriods(matt)

    # run model
    df_in = build_dataset(
        examples,
        region,
        model_afolu,
        regions,
        time_periods,
        **kwargs,
    )
    df_out = model_afolu(df_in)

    return df_in, df_out
