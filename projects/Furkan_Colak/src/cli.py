"""Single entry point: python -m src.cli <stage> --config <yaml>."""
import argparse
import importlib

from src.settings import load_config

STAGES = {
    "collect": "src.collect",
    "collect_ext": "src.collect_ext",
    "generate": "src.agents",
    "variants": "src.variants",
    "build": "src.dataset",
    "analyze": "src.features",
    "zeroshot": "src.zeroshot",
    "detect": "src.experiments",
    "evaluate": "src.evaluate",
    "calibrate": "src.calibration",
    "compare": "src.compare",
    "robust": "src.robust",
    "errors": "src.errors",
    "leakage": "src.leakage",
    "cost": "src.cost",
    "report": "src.report",
    "verify": "src.verify",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=STAGES)
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    module = importlib.import_module(STAGES[args.stage])
    module.run(load_config(args.config))


if __name__ == "__main__":
    main()
