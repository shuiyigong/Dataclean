from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from .filter_core import EpisodeFilterConfig, Stage1Config, Stage2Config, Stage3Config, Stage5Config
except ImportError:
    from filter_core import EpisodeFilterConfig, Stage1Config, Stage2Config, Stage3Config, Stage5Config


DEFAULT_CONFIG = Path(__file__).with_name("filter_config.json")


def load_filter_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def episode_filter_config_from_dict(raw: dict[str, Any]) -> EpisodeFilterConfig:
    stage1 = raw.get("stage1", {})
    stage2 = raw.get("stage2", {})
    stage3 = raw.get("stage3", {})
    stage5 = raw.get("stage5", {})
    return EpisodeFilterConfig(
        stage1=Stage1Config(
            enabled=stage1.get("enabled", Stage1Config.enabled),
            smooth_window=stage1.get("smooth_window", Stage1Config.smooth_window),
            median_window=stage1.get("median_window", Stage1Config.median_window),
            residual_mean_multiplier=stage1.get(
                "residual_mean_multiplier",
                Stage1Config.residual_mean_multiplier,
            ),
            min_threshold=stage1.get("min_threshold", Stage1Config.min_threshold),
        ),
        stage2=Stage2Config(
            enabled=stage2.get("enabled", Stage2Config.enabled),
            max_lag=stage2.get("max_lag", Stage2Config.max_lag),
            min_directional_agreement=stage2.get(
                "min_directional_agreement",
                Stage2Config.min_directional_agreement,
            ),
            state_dims=stage2.get("state_dims", None),
            action_dims=stage2.get("action_dims", None),
        ),
        stage3=Stage3Config(
            enabled=stage3.get("enabled", Stage3Config.enabled),
            lower_percentile=stage3.get("lower_percentile", Stage3Config.lower_percentile),
            upper_percentile=stage3.get("upper_percentile", Stage3Config.upper_percentile),
            alpha=stage3.get("alpha", Stage3Config.alpha),
            exempt_dims=stage3.get("exempt_dims", []),
        ),
        stage5=Stage5Config(
            enabled=stage5.get("enabled", Stage5Config.enabled),
            require_fixed_action_frame=stage5.get(
                "require_fixed_action_frame",
                Stage5Config.require_fixed_action_frame,
            ),
            action_frame=raw.get("action_frame", Stage5Config.action_frame),
        ),
        confidence_min=raw.get("confidence_min", EpisodeFilterConfig.confidence_min),
        max_bad_frame_ratio=raw.get("max_bad_frame_ratio", EpisodeFilterConfig.max_bad_frame_ratio),
    )


def filter_keys_from_dict(raw: dict[str, Any]) -> dict[str, str | None]:
    return {
        "action_key": raw.get("action_key", "action"),
        "state_key": raw.get("state_key"),
        "confidence_key": raw.get("confidence_key", "observation.confidence"),
    }


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Shared filter config JSON.")


def add_filter_override_args(parser: argparse.ArgumentParser, *, include_stage2: bool) -> None:
    parser.add_argument("--action-key", default=None)
    parser.add_argument("--confidence-key", default=None)
    parser.add_argument("--action-frame", default=None)
    parser.add_argument("--confidence-min", type=float, default=None)
    parser.add_argument("--max-bad-frame-ratio", type=float, default=None)
    parser.add_argument("--stage1-smooth-window", type=int, default=None)
    parser.add_argument("--stage1-median-window", type=int, default=None)
    parser.add_argument("--stage1-residual-mean-multiplier", type=float, default=None)
    parser.add_argument("--stage3-lower-percentile", type=float, default=None)
    parser.add_argument("--stage3-upper-percentile", type=float, default=None)
    parser.add_argument("--stage3-alpha", type=float, default=None)
    parser.add_argument("--stage3-exempt-dims", type=int, nargs="*", default=None)
    parser.add_argument("--disable-stage1", action="store_true")
    parser.add_argument("--disable-stage3", action="store_true")
    parser.add_argument("--disable-stage5", action="store_true")
    if include_stage2:
        parser.add_argument("--state-key", default=None)
        parser.add_argument("--enable-stage2", action="store_true")
        parser.add_argument("--stage2-max-lag", type=int, default=None)
        parser.add_argument("--stage2-min-directional-agreement", type=float, default=None)


def apply_filter_overrides(raw: dict[str, Any], args: argparse.Namespace, *, include_stage2: bool) -> dict[str, Any]:
    config = json.loads(json.dumps(raw))

    def set_if(name: str, value: Any) -> None:
        if value is not None:
            config[name] = value

    def set_stage_if(stage_name: str, field: str, value: Any) -> None:
        if value is not None:
            config.setdefault(stage_name, {})[field] = value

    set_if("action_key", args.action_key)
    set_if("confidence_key", args.confidence_key)
    set_if("action_frame", args.action_frame)
    set_if("confidence_min", args.confidence_min)
    set_if("max_bad_frame_ratio", args.max_bad_frame_ratio)

    set_stage_if("stage1", "smooth_window", args.stage1_smooth_window)
    set_stage_if("stage1", "median_window", args.stage1_median_window)
    set_stage_if("stage1", "residual_mean_multiplier", args.stage1_residual_mean_multiplier)
    set_stage_if("stage3", "lower_percentile", args.stage3_lower_percentile)
    set_stage_if("stage3", "upper_percentile", args.stage3_upper_percentile)
    set_stage_if("stage3", "alpha", args.stage3_alpha)
    set_stage_if("stage3", "exempt_dims", args.stage3_exempt_dims)

    if args.disable_stage1:
        config.setdefault("stage1", {})["enabled"] = False
    if args.disable_stage3:
        config.setdefault("stage3", {})["enabled"] = False
    if args.disable_stage5:
        config.setdefault("stage5", {})["enabled"] = False

    if include_stage2:
        set_if("state_key", args.state_key)
        set_stage_if("stage2", "max_lag", args.stage2_max_lag)
        set_stage_if("stage2", "min_directional_agreement", args.stage2_min_directional_agreement)
        if args.enable_stage2:
            config.setdefault("stage2", {})["enabled"] = True

    return config


def config_to_jsonable(config: EpisodeFilterConfig) -> dict[str, Any]:
    return asdict(config)
