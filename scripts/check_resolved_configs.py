from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from src.config import load_config

EXPECTED = {
    "dim10_eps2.yaml": {"dim": 10, "eps": 2.0, "baseline_n_samples": 50},
    "dim30_eps35.yaml": {"dim": 30, "eps": 3.5, "baseline_n_samples": 2000},
    "dim50_eps45.yaml": {"dim": 50, "eps": 4.5, "baseline_n_samples": 2000},
}


def validate(path: Path) -> dict:
    cfg = load_config(path)
    expected = EXPECTED[path.name]

    checks = {
        "dim": cfg.dim == expected["dim"],
        "eps_hit": cfg.metrics.eps_hit == expected["eps"],
        "training_100k": cfg.training.num_steps == 100_000,
        "v2_step_size": cfg.mode_finder_v2.step_size == 0.08,
        "v2_n_starts": cfg.mode_finder_v2.n_starts == 10,
        "v3f1_step_size": cfg.mode_finder_v3_f1.step_size == 0.08,
        "v3f2_step_size": cfg.mode_finder_v3_f2.step_size == 0.08,
        "v3f2_n_particles": cfg.mode_finder_v3_f2.n_particles == 150,
        "v3f2_timesteps": len(cfg.mode_finder_v3_f2.timesteps) == 13,
        "v3f2_ascent_steps": cfg.mode_finder_v3_f2.ascent_steps == 10,
        "v3f2_merge_factor": cfg.mode_finder_v3_f2.merge_factor == 1.0,
        "v3f2_merge_radius_min": cfg.mode_finder_v3_f2.merge_radius_min == 0.2,
        "v3f3_n_particles": cfg.mode_finder_v3_f3.n_particles == 150,
    }

    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise AssertionError(f"{path}: неверные параметры: {failed}")

    mf = cfg.mode_finder_v3_f2
    nfe_per_particle_max = (
        mf.ddim_steps + len(mf.timesteps) * mf.ascent_steps + mf.refine_steps
    )

    return {
        "path": str(path),
        "resolved": asdict(cfg),
        "derived": {
            "v3f2_nfe_per_particle_max": nfe_per_particle_max,
            "v3f2_nfe_default_population_max": nfe_per_particle_max * mf.n_particles,
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="configs/presets")
    parser.add_argument("--output", default="resolved_configs.json")
    args = parser.parse_args()

    config_dir = Path(args.config_dir)
    reports = [validate(config_dir / name) for name in EXPECTED]

    output = Path(args.output)
    output.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")

    for report in reports:
        r = report["resolved"]
        d = report["derived"]
        f2 = r["mode_finder_v3_f2"]
        print(
            f"OK {Path(report['path']).name}: dim={r['dim']}, eps={r['metrics']['eps_hit']}, "
            f"train={r['training']['num_steps']}, step={f2['step_size']}, "
            f"particles={f2['n_particles']}, levels={len(f2['timesteps'])}, "
            f"ascent={f2['ascent_steps']}, merge=({f2['merge_factor']}, {f2['merge_radius_min']}), "
            f"NFEmax/particle={d['v3f2_nfe_per_particle_max']}"
        )

    print(f"Полные resolved-конфигурации сохранены: {output}")


if __name__ == "__main__":
    main()
