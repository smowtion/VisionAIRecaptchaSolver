"""Phase 3 tests: class_mapping stays consistent with the solver's class taxonomy."""

from __future__ import annotations

import class_mapping as cm

from vision_ai_recaptcha_solver.types import CLASS_NAMES, TARGET_MAPPINGS


def test_folder_order_matches_model_class_ids() -> None:
    """FOLDER_ORDER must list folders in classification class-id order (alphabetical)."""
    assert sorted(cm.FOLDER_ORDER) == cm.FOLDER_ORDER
    assert len(cm.FOLDER_ORDER) == 14
    for idx, folder in enumerate(cm.FOLDER_ORDER):
        assert cm.FOLDER_TO_CLASS_ID[folder] == idx


def test_every_folder_maps_to_a_label() -> None:
    for folder in cm.FOLDER_ORDER:
        assert folder in cm.FOLDER_TO_LABEL
        assert cm.FOLDER_TO_LABEL[folder]


def test_target_folders_resolve_to_matching_class_id() -> None:
    """Each non-'Other' folder's label must resolve via TARGET_MAPPINGS to its class id."""
    for folder in cm.FOLDER_ORDER:
        if folder == "Other":
            continue
        label = cm.FOLDER_TO_LABEL[folder]
        assert label in TARGET_MAPPINGS, f"{label!r} missing from TARGET_MAPPINGS"
        assert TARGET_MAPPINGS[label] == cm.FOLDER_TO_CLASS_ID[folder]


def test_every_class_name_label_is_accounted_for() -> None:
    """Every solver label is a classification folder, a detection-only class, or an alias."""
    class_name_labels = [next(iter(entry)) for entry in CLASS_NAMES]
    for label in class_name_labels:
        accounted = (
            label in cm.LABEL_TO_FOLDER
            or label in cm.DETECTION_ONLY_LABELS
            or label in cm.LABEL_ALIASES
        )
        assert accounted, f"{label!r} is not mapped to any folder/alias/detection-only set"


def test_normalize_folder_is_stable() -> None:
    for folder in cm.FOLDER_ORDER:
        assert cm.normalize_folder(folder) == folder
        assert cm.normalize_folder(folder.lower()) == folder
        assert cm.normalize_folder(f"  {folder.upper()}  ") == folder


def test_normalize_folder_handles_known_variants() -> None:
    assert cm.normalize_folder("stair") == "Stair"
    assert cm.normalize_folder("traffic light") == "Traffic Light"
    assert cm.normalize_folder("hydrant") == "Hydrant"


def test_normalize_folder_unknown_raises() -> None:
    import pytest

    with pytest.raises(KeyError):
        cm.normalize_folder("not-a-real-class")


def test_validate_against_class_names_passes() -> None:
    # Should not raise: mapping is internally consistent with the solver taxonomy.
    cm.validate_against_class_names()
