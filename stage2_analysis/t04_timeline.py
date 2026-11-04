"""Table 4: the disclosed deployment timeline.

Not computed from the data. It is OpenAI's published record, transcribed into
data/deployment_timeline.csv, and it is what the observational series is read
AGAINST -- the paper does not use it to attribute anything. This script copies
it into output/ so every table in the paper has one artifact behind it, and
checks each date against the sample's coverage, because a disclosed event
falling in a corpus gap cannot be observed at all.

    output/table4_deployment_timeline.csv
"""
import os

import pandas as pd

from common import DATA, event_frame, write


def main():
    src = os.path.join(DATA, "deployment_timeline.csv")
    T = pd.read_csv(src, parse_dates=["date"])
    E = event_frame()

    covered = set(E.date.unique())
    T["turns_that_day"] = T.date.map(E.date.value_counts()).fillna(0).astype(int)
    T["observed"] = T.date.isin(covered)
    T["date"] = T.date.dt.date
    write(T, "table4_deployment_timeline.csv")

    gaps = T[~T.observed]
    print(f"  {len(T)} disclosed events, {len(gaps)} falling on days with no data")
    if len(gaps):
        # The rollback days are the important case: the corpus carries nothing
        # for April 28-30, so the rollback itself is not directly observable
        # and the post-rollback window necessarily starts on May 1.
        print(gaps[["date", "event"]].to_string(index=False))


if __name__ == "__main__":
    main()
