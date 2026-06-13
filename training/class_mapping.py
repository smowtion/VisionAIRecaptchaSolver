"""Single source of truth: dataset folder <-> classification class id <-> solver label.

The classification model (``recaptcha_classification_57k.onnx``) is trained on an
ImageFolder dataset whose folder names become class ids in **alphabetical** order. This
module pins that order and maps each folder to the English solver label used by
``vision_ai_recaptcha_solver.types`` so the dataset, the model, and the runtime never
drift apart.

Run ``python training/class_mapping.py`` to validate consistency against the solver.
"""

from __future__ import annotations

# Dataset folder names in classification class-id order (alphabetical == model order).
# 0 Bicycle 1 Bridge 2 Bus 3 Car 4 Chimney 5 Crosswalk 6 Hydrant
# 7 Motorcycle 8 Mountain 9 Other 10 Palm 11 Stair 12 Tractor 13 Traffic Light
FOLDER_ORDER: list[str] = [
    "Bicycle",
    "Bridge",
    "Bus",
    "Car",
    "Chimney",
    "Crosswalk",
    "Hydrant",
    "Motorcycle",
    "Mountain",
    "Other",
    "Palm",
    "Stair",
    "Tractor",
    "Traffic Light",
]

FOLDER_TO_CLASS_ID: dict[str, int] = {name: idx for idx, name in enumerate(FOLDER_ORDER)}

# Map each dataset folder to the English solver label (keys of types.CLASS_NAMES).
# "Other" is the negative/background class and has no solver target label.
FOLDER_TO_LABEL: dict[str, str] = {
    "Bicycle": "bicycles",
    "Bridge": "bridges",
    "Bus": "buses",
    "Car": "cars",
    "Chimney": "chimneys",
    "Crosswalk": "crosswalks",
    "Hydrant": "a fire hydrant",
    "Motorcycle": "motorcycles",
    "Mountain": "mountains or hills",
    "Other": "other",
    "Palm": "palm trees",
    "Stair": "stairs",
    "Tractor": "tractors",
    "Traffic Light": "traffic_lights",
}

# Reverse map (solver label -> folder), excluding the non-target "other" class.
LABEL_TO_FOLDER: dict[str, str] = {
    label: folder for folder, label in FOLDER_TO_LABEL.items() if folder != "Other"
}

# Solver labels that exist only for the 4x4 COCO detection model (no classification folder).
DETECTION_ONLY_LABELS: set[str] = {"boats", "parking meters"}

# Solver labels handled by an existing folder via alias (e.g. taxis are detected as cars).
LABEL_ALIASES: dict[str, str] = {"taxis": "Car"}

# Lowercased lookup for normalize_folder, including common singular/spacing variants.
_NORMALIZE_LOOKUP: dict[str, str] = {name.lower(): name for name in FOLDER_ORDER}
_NORMALIZE_LOOKUP.update(
    {
        "stair": "Stair",
        "stairs": "Stair",
        "trafficlight": "Traffic Light",
        "traffic_light": "Traffic Light",
        "hydrants": "Hydrant",
        "fire hydrant": "Hydrant",
        "palm tree": "Palm",
        "palms": "Palm",
    }
)


def normalize_folder(name: str) -> str:
    """Return the canonical dataset folder name for a raw/variant folder name.

    Args:
        name: Raw folder name (any case / surrounding whitespace / known variant).

    Returns:
        Canonical folder name from FOLDER_ORDER.

    Raises:
        KeyError: If the name does not map to a known class folder.
    """
    key = name.strip().lower()
    if key not in _NORMALIZE_LOOKUP:
        raise KeyError(f"Unknown class folder: {name!r}")
    return _NORMALIZE_LOOKUP[key]


def validate_against_class_names() -> None:
    """Assert the folder mapping is consistent with the solver class taxonomy.

    Raises:
        AssertionError: If a folder label is missing from TARGET_MAPPINGS, resolves to
            the wrong class id, or a solver label is unaccounted for.
    """
    from vision_ai_recaptcha_solver.types import CLASS_NAMES, TARGET_MAPPINGS

    for folder in FOLDER_ORDER:
        if folder == "Other":
            continue
        label = FOLDER_TO_LABEL[folder]
        assert label in TARGET_MAPPINGS, f"{label!r} missing from TARGET_MAPPINGS"
        assert TARGET_MAPPINGS[label] == FOLDER_TO_CLASS_ID[folder], (
            f"class id mismatch for {folder!r}: "
            f"{TARGET_MAPPINGS[label]} != {FOLDER_TO_CLASS_ID[folder]}"
        )

    for entry in CLASS_NAMES:
        label = next(iter(entry))
        assert (
            label in LABEL_TO_FOLDER or label in DETECTION_ONLY_LABELS or label in LABEL_ALIASES
        ), f"solver label {label!r} is unaccounted for in class_mapping"


if __name__ == "__main__":
    validate_against_class_names()
    print(f"class_mapping OK: {len(FOLDER_ORDER)} folders consistent with solver taxonomy")
