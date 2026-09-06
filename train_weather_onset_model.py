"""Train dry-antecedent rain-onset candidates with purged temporal validation.

Only rows dry for at least 60 minutes are used, while partition boundaries
match the main rain pipeline. Reports explicitly measure onset performance;
sensor-only onset models are candidates and are not automatically deployed.
"""

from train_weather_ai import main


if __name__ == "__main__":
    main(default_kind="rf_onset")
