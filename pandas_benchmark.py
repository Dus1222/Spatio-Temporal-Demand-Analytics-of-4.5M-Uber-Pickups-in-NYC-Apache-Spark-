"""Single-node pandas baseline for the same 'trips per hour' aggregation (for comparison with Spark)."""
import glob, json, os, statistics, sys, time
import pandas as pd

data = sys.argv[1] if len(sys.argv) > 1 else "data"
out = sys.argv[2] if len(sys.argv) > 2 else "out"
files = sorted(glob.glob(os.path.join(data, "uber-raw-data-*14.csv")))

def run():
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ts"] = pd.to_datetime(df["Date/Time"], format="%m/%d/%Y %H:%M:%S")
    return len(df), df.groupby(df["ts"].dt.hour).size()

runs = []
for _ in range(3):
    t0 = time.time(); n, g = run(); runs.append(time.time() - t0)
res = {"rows": int(n), "runs_s": [round(x, 2) for x in runs], "median_s": round(statistics.median(runs), 2),
       "peak_rss_mb": None,
       "pandas_version": pd.__version__}
json.dump(res, open(os.path.join(out, "pandas_benchmark.json"), "w"), indent=2)
print(res)
