from __future__ import annotations

from . import residual_diagnostics, rf_test, scenario_analysis


def main() -> None:
    print("Running scenario analysis...")
    scenario_analysis.main()
    print()

    print("Running random forest test...")
    rf_test.main()
    print()

    print("Running residual diagnostics...")
    residual_diagnostics.main()
    print()

    print("All diagnostics completed successfully.")
    print("Outputs written to:")
    print(" - model/artifacts/scenario_analysis/")
    print(" - model/artifacts/rf_test/")
    print(" - model/artifacts/residual_diagnostics/")


if __name__ == "__main__":
    main()
