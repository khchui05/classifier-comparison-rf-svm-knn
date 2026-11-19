# Data

Three binary classification problems from the [UCI Machine Learning Repository](https://archive.ics.uci.edu), loaded with the `ucimlrepo` package.

| Dataset | UCI id | Rows used | Features | Label (1 = …) | Positive rate |
|---|---|---|---|---|---|
| [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) | 222 | 10,000 (stratified subsample of 45,211) | 15 (6 numeric, 9 categorical) | client subscribed to a term deposit | 11.7% |
| [Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) | 186 | 6,497 | 11 (all numeric) | quality ≥ 7 ("good" wine) | 19.7% |
| [Student Performance](https://archive.ics.uci.edu/dataset/320/student+performance) | 320 | 649 | 30 (13 numeric, 17 categorical) | final grade G3 ≥ 10 (pass) | 84.6% |

## Preprocessing (done in the notebook)

- **Bank:** dropped `duration`. Call length is only known after the call ends, so it leaks the outcome. Missing categorical values were filled with `"unknown"`.
- **Student:** dropped the period grades `G1` and `G2`, which nearly determine `G3`.
- **All:** categorical columns are one-hot encoded. For SVM and KNN, numeric columns are also standardized. Both steps happen inside the pipeline, so they're fit on training folds only.

## Getting the data

The notebook downloads the data directly. To save local copies:

```bash
python modules/get_data.py
```

This writes `bank.csv`, `wine.csv` and `student.csv` into `data/00-raw/`.
