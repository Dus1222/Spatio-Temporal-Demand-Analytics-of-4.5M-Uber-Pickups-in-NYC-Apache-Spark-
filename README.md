# Spatio-Temporal Demand Analytics of 4.5 Million Uber Pickups in NYC using Apache Spark

Big Data Continuous Assessment Project | Manav Rachna University | Author: Dushyant

## Overview
A PySpark pipeline that cleans, stores and analyses 4,534,327 real Uber pickups in New York City
(April to September 2014). It finds demand patterns, identifies demand hotspots with K-Means and
forecasts hourly demand one day ahead with Spark MLlib.

## Dataset
FiveThirtyEight, "uber-tlc-foil-response" (originally from the NYC Taxi and Limousine Commission).
https://github.com/fivethirtyeight/uber-tlc-foil-response (folder `uber-trip-data`)
Six CSV files (`uber-raw-data-apr14.csv` to `uber-raw-data-sep14.csv`), about 210 MB.
Columns: `Date/Time`, `Lat`, `Lon`, `Base`. Downloaded automatically by `run_all.py`.

## Pipeline
1. Ingest the six CSV files with an explicit schema
2. Clean (remove 29,749 out-of-area records) and derive hour, weekday, month and km coordinates
3. Store as Parquet partitioned by month (about 10x smaller than CSV)
4. Spark SQL analytics by month, day, weekday, hour and base
5. K-Means hotspots (k chosen by silhouette score, giving 8 hotspots)
6. Day-ahead hourly demand forecast per hotspot (train April to August, test September)
7. Benchmark CSV vs Parquet vs pandas
8. Generate charts and tables from the pipeline output

## Key results (from the run used in the report)
| Item | Result |
|---|---|
| Records (raw / cleaned) | 4,534,327 / 4,504,578 |
| Demand growth, Apr to Sep | +82% pickups per day |
| Top two (Manhattan) hotspots | 76.8% of pickups |
| Best forecast (Linear Regression) | RMSE 79.26, MAE 36.79, R2 0.912 |
| Baseline (same hour last week) | RMSE 87.28, MAE 36.53, R2 0.893 |
| Parquet vs CSV | about 10x smaller, about 15x faster in Spark |

Random Forest and Gradient-Boosted Trees did not beat the baseline.

## Requirements
Python 3.10+, Java 17+, about 4 GB RAM.
```
python -m pip install pyspark==4.3.0 pandas matplotlib pillow
```

## How to run
Easiest (any computer): upload `Big_Data_Project_Colab.ipynb` to https://colab.research.google.com and choose Runtime > Run all.

On your own computer:
```
python run_all.py
```
This downloads the data into `data/` and runs the Spark pipeline, the pandas benchmark, the charts and the diagrams.
If Spark runs out of memory: `python run_all.py --driver-mem 3g`.

Step by step:
```
python code/uber_bigdata_pipeline.py --data data --out out
python code/pandas_benchmark.py data out
python code/make_figures.py out
python code/make_diagrams.py
```
Windows note: Spark can fail when writing Parquet with a `winutils` / `HADOOP_HOME` error. If so, use Google Colab or WSL.

## Project structure
```
Big_Data_Project/
  run_all.py                      one-command runner
  Big_Data_Project_Colab.ipynb    self-contained Colab notebook
  Big_Data_Project_Report.docx    project report
  Big_Data_Project_Presentation.pptx
  code/
    uber_bigdata_pipeline.py      Spark pipeline (ingest, clean, Parquet, SQL, K-Means, forecast, benchmark)
    pandas_benchmark.py           single-node pandas baseline
    make_figures.py               charts from pipeline output
    make_diagrams.py              workflow and architecture diagrams
  data/                           the six CSV files (downloaded)
  out/                            pipeline output (results.json, CSV tables, figures)
```

## Limitations
- Spark ran in local mode on one machine; no cluster speed-up is claimed. pandas was faster than Spark on CSV at this size.
- Data cover 2014 and Uber-affiliated bases only; no weather or event features.
- One test month, fixed hyper-parameters, no significance tests.
- The last hour of 30 September has no data (the file ends at 22:59), so 8 of 5,760 test rows are zero.

## SDG alignment
SDG 11 (sustainable cities and transport) and SDG 9 (data infrastructure and innovation). SDG 13 is an indirect link and was not measured.

## Reproducibility
Random seeds are fixed (42). Model scores should reproduce closely; timings depend on your hardware.

## References
FiveThirtyEight, uber-tlc-foil-response (GitHub, 2015). Zaharia et al., "Apache Spark: A unified engine for big data processing", Commun. ACM, 2016. Meng et al., "MLlib: Machine learning in Apache Spark", JMLR, 2016.
