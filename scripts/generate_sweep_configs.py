"""Generate concrete configs and manifest from a sweep specification."""

from __future__ import annotations

import argparse
from pathlib import Path

from Sandbox.singerclassifier.sweep import generate_sweep_configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sweep configs and manifest")
    parser.add_argument(
        "--sweep-spec",
        type=Path,
        required=True,
        help="Path to phase6_sweeps.yaml",
    )
    args = parser.parse_args()

    result = generate_sweep_configs(args.sweep_spec)

    print("Sweep generation complete")
    print("-------------------------")
    print(f"sweep_name: {result['sweep_name']}")
    print(f"generated configs: {result['num_configs']}")
    print(f"generated config dir: {result['generated_config_dir']}")
    print(f"manifest csv: {result['manifest_csv']}")
    print(f"manifest json: {result['manifest_json']}")
    print("\nCounts by family:")
    for family, count in result["family_counts"].items():
        print(f"  {family}: {count}")


if __name__ == "__main__":
    main()
