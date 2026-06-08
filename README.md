# 🏈 NFL Game Prediction Bot

A machine learning project that predicts NFL game outcomes using historical box score data, aggregated team statistics, and ELO ratings. The model outputs win probabilities for each matchup in a given week.

---

## Overview

This bot pulls real NFL game data week-by-week, builds rolling team performance averages (so predictions use only information available before each game), merges in ELO and QB ratings, and trains both a Logistic Regression and an XGBoost classifier to predict which team wins. Final output is a human-readable probability for each upcoming matchup.

---

## Features

- Scrapes live NFL box score data via `sportsipy`
- Builds week-over-week aggregated team stats (no data leakage — only past weeks used)
- Computes differential features between home and away teams
- Integrates FiveThirtyEight ELO ratings and QB value scores
- Trains and evaluates two models: Logistic Regression (scikit-learn) and XGBoost
- Prints win probabilities per matchup in plain English

---

## Requirements

Install dependencies with:

```bash
pip install sportsipy pandas numpy scikit-learn xgboost
```

Python 3.7+ is recommended.

---

## Data Sources

**Box Score Data** — scraped automatically via `sportsipy` from Pro Football Reference.

**ELO Ratings** — downloaded manually from FiveThirtyEight. Place the file at the path set in `get_elo()`:

```python
elo_df = pd.read_csv(r"path/to/nfl_elo_latest.csv")
```

You can download the latest ELO file here: https://github.com/fivethirtyeight/data/tree/master/nfl-elo

> **Note:** Update the file path in `get_elo()` to match your local setup before running.

---

## How It Works

### Data Pipeline

1. **`get_schedule(year)`** — Fetches the full season schedule for a given year (weeks 1–17).
2. **`game_data_up_to_week(weeks, year)`** — Pulls box score stats for each game across the specified weeks.
3. **`agg_weekly_data(...)`** — Aggregates each team's stats from all *prior* weeks before a matchup. Computes differential features (e.g., `pass_yards_dif = away_pass_yards - home_pass_yards`) for every stat category.
4. **`get_elo()`** — Loads and cleans the FiveThirtyEight ELO dataset, mapping team abbreviations to match `sportsipy` formatting.
5. **`merge_rankings(...)`** — Merges ELO and QB value differentials into the game feature set.
6. **`prep_test_train(current_week, weeks, year)`** — Orchestrates the full pipeline and returns a training set (completed games) and a test/prediction set (current week's games).

### Features Used

Each game row contains ~40 differential stats between the away and home team, including:

- Win percentage, points scored, total yards
- Passing (yards, TDs, completions, attempts, sacks)
- Rushing (yards, TDs, attempts)
- Turnovers (interceptions, fumbles)
- 3rd and 4th down conversion rates
- Time of possession, penalties, yards from penalties
- ELO rating differential
- QB value differential

### Models

**Logistic Regression** (scikit-learn)
- L1 penalty (Lasso) with `liblinear` solver
- `class_weight='balanced'` to handle any class imbalance
- Outputs win probabilities via `predict_proba`

**XGBoost**
- Linear booster (`gblinear`) with shuffle feature selector
- Trained for 1,000 rounds with train/test eval logging
- Binary classification objective

---

## Usage

Set your target week and year at the bottom of the script:

```python
current_week = 8          # Week you want to predict
weeks = list(range(1, current_week + 1))
year = 2021               # NFL season year
```

Then run the script. It will:

1. Pull all game data through `current_week`
2. Train both models on completed games
3. Print win probabilities for each game in the current week

**Example output:**
```
The Las Vegas Raiders have a probability of 0.63 of beating the New York Giants.
The Tampa Bay Buccaneers have a probability of 0.71 of beating the Chicago Bears.
```

---

## Project Structure

```
NFL_Prediction.py       # Main script — data pipeline, models, and output
nfl_elo_latest.csv      # ELO ratings file (download separately, see above)
README.md
```

---

## Known Limitations & Notes

- The ELO file path in `get_elo()` is hardcoded — update it to your local path before running.
- `sportsipy` scrapes Pro Football Reference and can be slow for large date ranges; expect runtime of several minutes when pulling a full season.
- Predictions are only as good as the data available — early-season weeks (1–2) have very limited historical averages to train on.
- The ELO data in this version is filtered through early January 2022 (`< '01-05-2022'`); update this filter for future seasons.
- XGBoost is trained in this script but final predictions use the Logistic Regression model — you can swap in the XGBoost model (`bst`) for comparison.

---

## Future Improvements

- Add support for playoff weeks (weeks 18+)
- Persist trained models with `joblib` or `pickle` to avoid retraining each run
- Pull updated ELO data automatically instead of via manual CSV download
- Add a command-line interface (CLI) for easier week/year configuration
- Expand to include injury reports or weather data as features