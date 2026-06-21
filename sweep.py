"""Controlled sweep specification expansion and manifest generation."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import re
from pathlib import Path
from typing import Any

import yaml

AUGMENTATION_PROFILES: dict[str, dict[str, Any]] = {
    "light": {
        "enabled": True,
        "waveform_noise_std": 0.001,
        "time_mask": {"enabled": True, "max_width": 25, "num_masks": 1},
        "freq_mask": {"enabled": True, "max_width": 6, "num_masks": 1},
    },
    "medium": {
        "enabled": True,
        "waveform_noise_std": 0.003,
        "time_mask": {"enabled": True, "max_width": 40, "num_masks": 1},
        "freq_mask": {"enabled": True, "max_width": 8, "num_masks": 1},
    },
}

FAMILY_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "majority_baseline": {
        "main_change": "Predict most frequent train class",
        "notes": "Trivial reference point",
    },
    "cnn_basic": {
        "main_change": "Scratch CNN over log-mel spectrograms",
        "notes": "First valid deep-learning model",
    },
    "cnn_balanced": {
        "main_change": "+ class-weighted cross entropy",
        "notes": "Tests imbalance-aware training",
    },
    "cnn_augmented": {
        "main_change": "+ light/medium augmentation",
        "notes": "Tests generalization from augmentation",
    },
    "cnn_augmented_multicrop": {
        "main_change": "+ multi-crop evaluation",
        "notes": "Tests recording-level prediction stability",
    },
}


def load_sweep_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    current = config
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def _format_token(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text.replace(".", "p")
    if isinstance(value, int):
        return str(value)
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", "", text)
    return text or "val"


def _run_name_tokens(family: str, sweep_params: dict[str, Any]) -> list[str]:
    tokens = [family]
    key_order = [
        ("training.learning_rate", "lr"),
        ("model.dropout", "drop"),
        ("training.weight_decay", "wd"),
        ("augmentation.profile", "aug"),
        ("data.eval_num_crops", "crop"),
    ]
    for dotted_key, prefix in key_order:
        if dotted_key in sweep_params:
            tokens.append(f"{prefix}{_format_token(sweep_params[dotted_key])}")
    for key, value in sorted(sweep_params.items()):
        if any(key == item[0] for item in key_order):
            continue
        short = key.split(".")[-1]
        tokens.append(f"{short}{_format_token(value)}")
    return tokens


def generate_run_name(family: str, sweep_params: dict[str, Any]) -> str:
    if family == "majority_baseline":
        return "majority_baseline"
    return "_".join(_run_name_tokens(family, sweep_params))


def resolve_augmentation_cfg(config: dict[str, Any]) -> dict[str, Any] | None:
    data_cfg = config.get("data", {})
    # Support misplaced top-level augment_train for backward compatibility.
    augment_train = bool(
        data_cfg.get("augment_train", config.get("augment_train", False))
    )
    if not augment_train:
        return None

    aug_cfg = copy.deepcopy(config.get("augmentation", {}))
    profile = aug_cfg.pop("profile", None)
    if profile:
        profile_cfg = AUGMENTATION_PROFILES.get(str(profile))
        if profile_cfg is None:
            raise ValueError(f"Unknown augmentation profile: {profile}")
        # Profile settings must override sweep defaults such as enabled: false.
        aug_cfg = deep_merge(aug_cfg, profile_cfg)

    if not aug_cfg.get("enabled", True):
        return None

    time_mask = aug_cfg.get("time_mask") or {}
    freq_mask = aug_cfg.get("freq_mask") or {}
    has_noise = float(aug_cfg.get("waveform_noise_std", 0.0)) > 0.0
    has_time_mask = bool(time_mask.get("enabled", False))
    has_freq_mask = bool(freq_mask.get("enabled", False))
    if not (has_noise or has_time_mask or has_freq_mask):
        raise ValueError(
            "augment_train=true but resolved augmentation has no active operations: "
            f"{aug_cfg}"
        )

    return aug_cfg


def get_augmentation_profile_name(config: dict[str, Any]) -> str:
    """Return the configured profile name, or none/custom."""
    data_cfg = config.get("data", {})
    augment_train = bool(
        data_cfg.get("augment_train", config.get("augment_train", False))
    )
    if not augment_train:
        return "none"

    profile = config.get("augmentation", {}).get("profile")
    if profile:
        return str(profile)
    return "custom"


def build_experiment_config(
    spec: dict[str, Any],
    family_key: str,
    experiment_def: dict[str, Any],
    sweep_params: dict[str, Any],
) -> dict[str, Any]:
    family = experiment_def.get("family", family_key)
    run_name = generate_run_name(family, sweep_params)

    config = {
        "run_name": run_name,
        "family": family,
        "sweep_name": spec["sweep_name"],
        "seed": spec.get("seed", 42),
        "sweep_parameters": sweep_params,
        "data": deep_merge(
            {"split_csv": spec["paths"]["split_csv"]},
            spec.get("defaults", {}).get("data", {}),
        ),
        "model": copy.deepcopy(spec.get("defaults", {}).get("model", {})),
        "training": copy.deepcopy(spec.get("defaults", {}).get("training", {})),
        "augmentation": copy.deepcopy(spec.get("defaults", {}).get("augmentation", {})),
        "audio_cache": copy.deepcopy(spec.get("defaults", {}).get("audio_cache", {})),
        "output": {"root_dir": spec["paths"]["output_root"]},
    }

    fixed = experiment_def.get("fixed", {})
    config = deep_merge(config, fixed)

    for dotted_key, value in sweep_params.items():
        set_nested(config, dotted_key, value)

    exp_type = experiment_def.get("type", "train_eval")
    if exp_type == "baseline":
        config["model"] = deep_merge(config.get("model", {}), {"type": "majority_class"})

    return config


def expand_experiment_variants(
    spec: dict[str, Any],
    family_key: str,
    experiment_def: dict[str, Any],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    sweep = experiment_def.get("sweep", {})
    if not sweep:
        return [(build_experiment_config(spec, family_key, experiment_def, {}), {})]

    keys = list(sweep.keys())
    values = [sweep[key] if isinstance(sweep[key], list) else [sweep[key]] for key in keys]
    variants: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for combo in itertools.product(*values):
        sweep_params = dict(zip(keys, combo))
        config = build_experiment_config(spec, family_key, experiment_def, sweep_params)
        variants.append((config, sweep_params))

    return variants


def generate_sweep_configs(spec_path: str | Path) -> dict[str, Any]:
    spec = load_sweep_spec(spec_path)
    generated_dir = Path(spec["paths"]["generated_config_dir"])
    generated_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}
    index = 0

    for family_key, experiment_def in spec.get("experiments", {}).items():
        family = experiment_def.get("family", family_key)
        exp_type = experiment_def.get("type", "train_eval")
        variants = expand_experiment_variants(spec, family_key, experiment_def)
        family_counts[family] = family_counts.get(family, 0) + len(variants)

        for config, _ in variants:
            run_name = config["run_name"]
            config_path = generated_dir / f"{run_name}.yaml"
            with config_path.open("w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            run_dir = Path(spec["paths"]["output_root"]) / run_name
            manifest_rows.append(
                {
                    "index": index,
                    "sweep_name": spec["sweep_name"],
                    "family": family,
                    "run_name": run_name,
                    "type": exp_type,
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "status": "pending",
                }
            )
            index += 1

    manifest_csv = Path(spec["paths"]["manifest_csv"])
    manifest_json = Path(spec["paths"]["manifest_json"])
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "index",
        "sweep_name",
        "family",
        "run_name",
        "type",
        "config_path",
        "run_dir",
        "status",
    ]
    with manifest_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    manifest_payload = {
        "sweep_name": spec["sweep_name"],
        "spec_path": str(spec_path),
        "generated_config_dir": str(generated_dir),
        "runs": manifest_rows,
    }
    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2)

    validate_manifest_configs(manifest_rows)

    return {
        "sweep_name": spec["sweep_name"],
        "num_configs": len(manifest_rows),
        "family_counts": family_counts,
        "manifest_csv": str(manifest_csv),
        "manifest_json": str(manifest_json),
        "generated_config_dir": str(generated_dir),
        "manifest_rows": manifest_rows,
    }


def validate_manifest_configs(manifest_rows: list[dict[str, Any]]) -> None:
    missing = [
        row["config_path"]
        for row in manifest_rows
        if not Path(row["config_path"]).is_file()
    ]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "Generated manifest references missing config files:\n" + lines
        )


def read_manifest_csv(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["index"] = int(row["index"])
    return rows


def write_manifest_csv(manifest_path: str | Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "index",
        "sweep_name",
        "family",
        "run_name",
        "type",
        "config_path",
        "run_dir",
        "status",
    ]
    with Path(manifest_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = Path(manifest_path).with_suffix(".json")
    if json_path.is_file():
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["runs"] = rows
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
