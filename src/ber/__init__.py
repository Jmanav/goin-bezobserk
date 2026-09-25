"""Business entity resolution pipeline -- Owner A foundations (Sprint 0).

Public surface for Owners B, C and D. The normalised-field contract is
`normalise.NORMALISED_FIELDS` (task A2.1); code against that list rather than
reaching into the modules.
"""

from .safe_io import (  # noqa: F401
    EXPECTED_COLUMNS,
    SanityCheckError,
    SanityReport,
    check_ground_truth_ids,
    check_namespace,
    check_source,
    load_source,
    read_source,
)
from .normalise import (  # noqa: F401
    NORMALISED_FIELDS,
    Normalised,
    fold_ascii,
    normalise_address,
    normalise_country,
    normalise_frame,
    normalise_name,
    normalise_record,
)
from .placeholders import (  # noqa: F401
    discover_placeholders,
    is_placeholder,
    missing_flags,
)

__all__ = [
    "EXPECTED_COLUMNS",
    "NORMALISED_FIELDS",
    "Normalised",
    "SanityCheckError",
    "SanityReport",
    "check_ground_truth_ids",
    "check_namespace",
    "check_source",
    "discover_placeholders",
    "fold_ascii",
    "is_placeholder",
    "load_source",
    "missing_flags",
    "normalise_address",
    "normalise_country",
    "normalise_frame",
    "normalise_name",
    "normalise_record",
    "read_source",
]

from . import decode, scorer, submission  # noqa: E402,F401
from . import blocking, blocking_report, dense  # noqa: E402,F401
from . import features, matcher  # noqa: E402,F401
