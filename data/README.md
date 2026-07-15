# data/

Place the `NewspaperChurn.csv` dataset in this folder before running the
pipeline:

```
data/NewspaperChurn.csv
```

The file is intentionally not committed to this repository (see
`.gitignore`) - only its expected location is documented here. The
pipeline reads it via `src/config.py`'s `RAW_DATA_PATH`.

Expected schema: one row per subscriber, including a `Subscriber` column
with values `YES`/`NO`, an identifier column, several geographic columns
(dropped before modelling), and the demographic/account features used as
predictors. See `src/config.py` and `src/preprocessing.py` for the exact
column list and encoding.
