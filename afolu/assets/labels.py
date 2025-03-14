import dagster as dg
import numpy as np

from afolu.assets.constants import LABEL_LIST


# pylint: disable=no-value-for-parameter
@dg.asset(io_manager_key="json_manager")
def transition_label_map() -> dict[int, list[str, str]]:
    multiplier = 10 ** np.ceil(np.log10(len(LABEL_LIST)))

    out = {}
    for i, start_label in enumerate(LABEL_LIST):
        for j, end_label in enumerate(LABEL_LIST):
            key = int(i * multiplier + j)
            assert key not in out
            out[key] = [start_label, end_label]

    return out
