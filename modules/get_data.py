"""Download the three UCI datasets used in this project into data/00-raw/.

Run from the repository root:
    python modules/get_data.py
"""
import os
from ucimlrepo import fetch_ucirepo

DATASETS = {"bank": 222, "wine": 186, "student": 320}


def get_raw(destination_directory=os.path.join("data", "00-raw")):
    os.makedirs(destination_directory, exist_ok=True)
    for name, uci_id in DATASETS.items():
        d = fetch_ucirepo(id=uci_id)
        path = os.path.join(destination_directory, f"{name}.csv")
        d.data.original.to_csv(path, index=False)
        print(f"Downloaded {name} (UCI id {uci_id}) -> {path}")


if __name__ == "__main__":
    get_raw()
