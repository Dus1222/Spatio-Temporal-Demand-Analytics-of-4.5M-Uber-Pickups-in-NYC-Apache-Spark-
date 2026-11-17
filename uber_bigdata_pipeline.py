"""
Spatio-Temporal Demand Analytics of 4.5 Million Uber Pickups in New York City
Big Data Continuous Assessment Project  |  Manav Rachna University

End-to-end PySpark pipeline:
  1. Ingestion      (6 monthly CSV files -> Spark DataFrame, explicit schema)
  2. Cleaning       (timestamp parsing, null / out-of-area filtering, data-quality report)
  3. Storage        (cleaned data written as Parquet, partitioned by month)
  4. Analytics      (Spark SQL: monthly / hourly / weekday / base-level demand)
  5. Geo-clustering (MLlib K-Means on pickup coordinates -> demand hotspots)
  6. Forecasting    (MLlib regression: day-ahead hourly demand per hotspot)
  7. Benchmarks     (CSV vs Parquet vs partition-pruned read)

Usage:
    python uber_bigdata_pipeline.py --data data --out out [--master local[*]]

Every number reported in the project report/presentation is produced by this script
(see out/results.json and out/*.csv).
"""
import argparse
import glob
import json
import math
import os
import shutil
import statistics
import time

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator, RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder, VectorAssembler
from pyspark.ml.regression import GBTRegressor, LinearRegression, RandomForestRegressor
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

# ----------------------------------------------------------------------------- config
NYC_LAT = (40.49, 40.92)      # bounding box of the NYC metro pickup area
NYC_LON = (-74.27, -73.68)
CANDIDATE_K = [6, 8, 10, 12, 15]   # declared before running; best silhouette is chosen
SEED = 42
TEST_START = "2014-09-01"          # temporal split: train Apr-Aug, test September

LANDMARKS = {  # used only to give hotspots a readable name (nearest landmark within 3 km)
    "Times Square": (40.7580, -73.9855), "Penn Station": (40.7506, -73.9935),
    "Grand Central": (40.7527, -73.9772), "Financial District": (40.7074, -74.0113),
    "Union Square": (40.7359, -73.9911), "Greenwich Village": (40.7335, -74.0027),
    "Upper East Side": (40.7736, -73.9566), "Upper West Side": (40.7870, -73.9754),
    "Harlem": (40.8116, -73.9465), "Williamsburg": (40.7081, -73.9571),
    "Downtown Brooklyn": (40.6928, -73.9903), "LaGuardia Airport": (40.7769, -73.8740),
    "JFK Airport": (40.6413, -73.7781), "Long Island City": (40.7447, -73.9485),
    "Astoria": (40.7644, -73.9235), "Park Slope": (40.6710, -73.9777),
    "Jersey City (NJ)": (40.7178, -74.0431), "Newark Airport (EWR)": (40.6895, -74.1745), "Bronx (Fordham)": (40.8618, -73.8901),
}

# local-tangent-plane projection so K-Means distance is in kilometres
KM_PER_DEG_LAT = 111.0
KM_PER_DEG_LON = 111.32 * math.cos(math.radians(40.7))   # ~84.4 km at NYC latitude


def haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(h))


def nearest_landmark(lat, lon, max_km=3.0):
    name, d = min(((n, haversine_km((lat, lon), c)) for n, c in LANDMARKS.items()), key=lambda x: x[1])
    return name if d <= max_km else "Other area"


def timed(label, store, fn):
    t0 = time.time()
    out = fn()
    store[label] = round(time.time() - t0, 2)
    print(f"[time] {label}: {store[label]} s", flush=True)
    return out


def dir_size_mb(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 1e6, 2)


def rows_to_dicts(rows):
    return [r.asDict() for r in rows]


def write_csv(rows, path, cols):
    import csv
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    ap.add_argument("--master", default="local[*]")
    ap.add_argument("--driver-mem", default="1500m")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    spark = (SparkSession.builder.master(args.master).appName("UberDemandAnalytics")
             .config("spark.driver.memory", args.driver_mem)
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.sql.ansi.enabled", "false")
             .config("spark.ui.showConsoleProgress", "false").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    R = {"env": {"spark_version": spark.version, "master": args.master,
                 "default_parallelism": spark.sparkContext.defaultParallelism,
                 "cpu_count": os.cpu_count(), "driver_memory": args.driver_mem},
         "timings_s": {}}
    T = R["timings_s"]

    # ------------------------------------------------------------------ 1. INGESTION
    schema = StructType([StructField("ts_raw", StringType()), StructField("Lat", DoubleType()),
                         StructField("Lon", DoubleType()), StructField("Base", StringType())])
    csv_files = sorted(glob.glob(os.path.join(args.data, "uber-raw-data-*14.csv")))
    R["input_files"] = [os.path.basename(f) for f in csv_files]
    R["input_csv_size_mb"] = round(sum(os.path.getsize(f) for f in csv_files) / 1e6, 1)
    raw = spark.read.csv(csv_files, header=True, schema=schema)
    raw_n = timed("ingest_count_raw", T, lambda: raw.count())
    R["raw_rows"] = raw_n

    # ------------------------------------------------------------------ 2. CLEANING
    parsed = raw.withColumn("pickup_ts", F.to_timestamp("ts_raw", "M/d/yyyy H:mm:ss"))
    dq = parsed.agg(
        F.sum(F.col("pickup_ts").isNull().cast("int")).alias("null_ts"),
        F.sum(F.col("Lat").isNull().cast("int")).alias("null_lat"),
        F.sum(F.col("Lon").isNull().cast("int")).alias("null_lon"),
        F.sum(F.col("Base").isNull().cast("int")).alias("null_base"),
        F.sum((~F.col("Lat").between(*NYC_LAT) | ~F.col("Lon").between(*NYC_LON)).cast("int")).alias("outside_nyc"),
    ).collect()[0].asDict()
    R["data_quality"] = {k: int(v or 0) for k, v in dq.items()}

    clean = (parsed.filter(F.col("pickup_ts").isNotNull() & F.col("Lat").between(*NYC_LAT)
                           & F.col("Lon").between(*NYC_LON) & F.col("Base").isNotNull())
             .withColumn("hour", F.hour("pickup_ts"))
             .withColumn("dow", (F.dayofweek("pickup_ts") + 5) % 7)          # Mon=0 .. Sun=6
             .withColumn("is_weekend", (F.col("dow") >= 5).cast("int"))
             .withColumn("month", F.month("pickup_ts"))
             .withColumn("date", F.to_date("pickup_ts"))
             .withColumn("y_km", (F.col("Lat") - 40.7) * KM_PER_DEG_LAT)
             .withColumn("x_km", (F.col("Lon") + 74.0) * KM_PER_DEG_LON)
             .drop("ts_raw"))
    clean.persist(StorageLevel.DISK_ONLY)
    clean_n = timed("clean_count", T, lambda: clean.count())
    R["clean_rows"] = clean_n
    R["rows_removed"] = raw_n - clean_n
    R["rows_removed_pct"] = round(100 * (raw_n - clean_n) / raw_n, 4)

    distinct_n = timed("distinct_count", T, lambda: clean.select("pickup_ts", "Lat", "Lon", "Base").distinct().count())
    R["exact_duplicate_rows"] = clean_n - distinct_n
    R["exact_duplicate_pct"] = round(100 * (clean_n - distinct_n) / clean_n, 3)

    span = clean.agg(F.min("pickup_ts").alias("mn"), F.max("pickup_ts").alias("mx"),
                     F.countDistinct("date").alias("days"), F.countDistinct("Base").alias("bases"),
                     F.min("Lat").alias("lat_min"), F.max("Lat").alias("lat_max"),
                     F.min("Lon").alias("lon_min"), F.max("Lon").alias("lon_max")).collect()[0].asDict()
    R["span"] = {k: str(v) for k, v in span.items()}

    # ------------------------------------------------------------------ 3. STORAGE (Parquet)
    pq_path = os.path.join(args.out, "uber_clean.parquet")
    shutil.rmtree(pq_path, ignore_errors=True)
    timed("write_parquet", T, lambda: clean.drop("y_km", "x_km").write.partitionBy("month").parquet(pq_path))
    R["parquet_size_mb"] = dir_size_mb(pq_path)
    R["compression_ratio_csv_to_parquet"] = round(R["input_csv_size_mb"] / R["parquet_size_mb"], 2)

    clean.unpersist()
    # ------------------------------------------------------------------ 4. SPARK SQL ANALYTICS
    # From here on every stage reads the cleaned Parquet layer (not the raw CSV files).
    clean = (spark.read.parquet(pq_path)
             .withColumn("y_km", (F.col("Lat") - 40.7) * KM_PER_DEG_LAT)
             .withColumn("x_km", (F.col("Lon") + 74.0) * KM_PER_DEG_LON))
    clean.createOrReplaceTempView("trips")

    def sql(q):
        return spark.sql(q).collect()

    monthly = timed("q_monthly", T, lambda: sql(
        "SELECT month, COUNT(*) AS trips, COUNT(DISTINCT date) AS days, "
        "ROUND(COUNT(*)/COUNT(DISTINCT date),0) AS trips_per_day FROM trips GROUP BY month ORDER BY month"))
    write_csv(monthly, f"{args.out}/q_monthly.csv", ["month", "trips", "days", "trips_per_day"])

    hourly = timed("q_hourly_weekday_weekend", T, lambda: sql(
        "SELECT hour, is_weekend, ROUND(COUNT(*)/COUNT(DISTINCT date),1) AS avg_trips_per_day "
        "FROM trips GROUP BY hour, is_weekend ORDER BY is_weekend, hour"))
    write_csv(hourly, f"{args.out}/q_hourly.csv", ["hour", "is_weekend", "avg_trips_per_day"])

    heat = timed("q_heatmap", T, lambda: sql(
        "SELECT t.dow, t.hour, ROUND(t.n / d.ndays, 1) AS avg_trips "
        "FROM (SELECT dow, hour, COUNT(*) AS n FROM trips GROUP BY dow, hour) t "
        "JOIN (SELECT dow, COUNT(DISTINCT date) AS ndays FROM trips GROUP BY dow) d ON t.dow = d.dow "
        "ORDER BY t.dow, t.hour"))
    write_csv(heat, f"{args.out}/q_heatmap.csv", ["dow", "hour", "avg_trips"])

    daily = timed("q_daily", T, lambda: sql(
        "SELECT CAST(date AS STRING) AS date, dow, COUNT(*) AS trips FROM trips GROUP BY date, dow ORDER BY date"))
    write_csv(daily, f"{args.out}/q_daily.csv", ["date", "dow", "trips"])

    base = timed("q_base", T, lambda: sql(
        "SELECT Base, month, COUNT(*) AS trips FROM trips GROUP BY Base, month ORDER BY Base, month"))
    write_csv(base, f"{args.out}/q_base_month.csv", ["Base", "month", "trips"])

    # ------------------------------------------------------------------ 5. K-MEANS HOTSPOTS
    assembler = VectorAssembler(inputCols=["x_km", "y_km"], outputCol="features")
    feats = assembler.transform(clean.select("Lat", "Lon", "x_km", "y_km", "date", "hour"))
    feats.persist(StorageLevel.DISK_ONLY)
    sample = feats.sample(False, 0.05, SEED).persist(StorageLevel.MEMORY_AND_DISK)
    R["silhouette_sample_rows"] = sample.count()
    sil = {}
    ev = ClusteringEvaluator(featuresCol="features", predictionCol="prediction", metricName="silhouette")

    def fit_sil():
        for k in CANDIDATE_K:
            m = KMeans(k=k, seed=SEED, maxIter=30, featuresCol="features").fit(sample)
            sil[k] = round(ev.evaluate(m.transform(sample)), 4)
    timed("kmeans_silhouette_search", T, fit_sil)
    best_k = max(sil, key=sil.get)
    R["silhouette_by_k"] = sil
    R["best_k"] = best_k
    sample.unpersist()

    km_model = timed("kmeans_final_fit_full_data", T,
                     lambda: KMeans(k=best_k, seed=SEED, maxIter=40, featuresCol="features").fit(feats))
    R["kmeans_training_cost_km2"] = round(km_model.summary.trainingCost, 1)
    centers = []
    for i, c in enumerate(km_model.clusterCenters()):
        lat = float(40.7 + c[1] / KM_PER_DEG_LAT)
        lon = float(-74.0 + c[0] / KM_PER_DEG_LON)
        centers.append({"zone": int(i), "lat": round(lat, 5), "lon": round(lon, 5), "landmark": nearest_landmark(lat, lon)})

    assigned = (km_model.transform(feats).withColumnRenamed("prediction", "zone")
                .select("Lat", "Lon", "date", "hour", "zone"))
    zone_sizes = {r["zone"]: r["n"] for r in assigned.groupBy("zone").agg(F.count("*").alias("n")).collect()}
    for c in centers:
        c["trips"] = int(zone_sizes[c["zone"]])
        c["share_pct"] = round(100 * c["trips"] / clean_n, 2)
    centers.sort(key=lambda c: -c["trips"])
    R["hotspots"] = centers
    write_csv(centers, f"{args.out}/hotspots.csv", ["zone", "landmark", "lat", "lon", "trips", "share_pct"])

    # sample of labelled points for the map figure
    pts = assigned.sample(False, 0.03, SEED).select("Lat", "Lon", "zone").collect()
    write_csv(pts, f"{args.out}/map_sample.csv", ["Lat", "Lon", "zone"])

    # ------------------------------------------------------------------ 6. DEMAND FORECASTING
    hourly_zone = assigned.groupBy("date", "hour", "zone").agg(F.count("*").alias("cnt")).persist(StorageLevel.MEMORY_AND_DISK)
    hourly_zone.count()
    feats.unpersist()
    dates = hourly_zone.select("date").distinct()
    grid = (dates.crossJoin(spark.range(24).select(F.col("id").cast("int").alias("hour")))
            .crossJoin(spark.range(best_k).select(F.col("id").cast("int").alias("zone"))))
    full = (grid.join(hourly_zone, ["date", "hour", "zone"], "left").fillna({"cnt": 0})
            .withColumn("dow", (F.dayofweek("date") + 5) % 7)
            .withColumn("is_weekend", (F.col("dow") >= 5).cast("int")))
    zc = spark.createDataFrame([(c["zone"], c["lat"], c["lon"]) for c in centers], ["zone", "zlat", "zlon"])
    full = full.join(zc, "zone")
    w = Window.partitionBy("zone").orderBy("date", "hour")
    full = (full.withColumn("lag_24", F.lag("cnt", 24).over(w))
            .withColumn("lag_168", F.lag("cnt", 168).over(w))
            .dropna(subset=["lag_24", "lag_168"]).withColumn("label", F.col("cnt").cast("double")))
    full.persist(StorageLevel.MEMORY_AND_DISK)
    train = full.filter(F.col("date") < TEST_START)
    test = full.filter(F.col("date") >= TEST_START)
    R["forecast_rows"] = {"total": full.count(), "train": train.count(), "test": test.count()}

    # diagnostic: does the test month fall outside the range seen in training? (tree models cannot extrapolate)
    st = {}
    for nm, d in (("train", train), ("test", test)):
        r = d.agg(F.avg("label").alias("mean_demand"), F.max("label").alias("max_demand"),
                  F.max("lag_168").alias("max_lag168"), F.avg("lag_168").alias("mean_lag168")).collect()[0].asDict()
        st[nm] = {k: round(float(v), 2) for k, v in r.items()}
    st["test_rows_with_lag168_above_train_max"] = test.filter(F.col("lag_168") > st["train"]["max_lag168"]).count()
    st["test_rows_with_label_above_train_max"] = test.filter(F.col("label") > st["train"]["max_demand"]).count()
    R["split_stats"] = st

    ohe = OneHotEncoder(inputCols=["zone"], outputCols=["zone_ohe"], dropLast=False)
    asm = VectorAssembler(inputCols=["hour", "dow", "is_weekend", "zlat", "zlon", "lag_24", "lag_168", "zone_ohe"],
                          outputCol="features")
    rmse_ev = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
    mae_ev = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
    r2_ev = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="r2")

    def score(df):
        return {"rmse": round(rmse_ev.evaluate(df), 2), "mae": round(mae_ev.evaluate(df), 2),
                "r2": round(r2_ev.evaluate(df), 4)}

    metrics = {}
    # baseline 1: same hour, same zone, one week earlier
    metrics["Baseline: same slot last week"] = score(test.withColumn("prediction", F.col("lag_168").cast("double")))
    # baseline 2: training-set mean per (zone, hour, weekday)
    prof = train.groupBy("zone", "hour", "dow").agg(F.avg("label").alias("prediction"))
    metrics["Baseline: mean by zone-hour-weekday"] = score(test.drop("prediction").join(prof, ["zone", "hour", "dow"], "left").fillna({"prediction": 0.0}))

    models = {
        "Linear Regression": LinearRegression(featuresCol="features", labelCol="label", regParam=0.1),
        "Random Forest": RandomForestRegressor(featuresCol="features", labelCol="label", numTrees=60, maxDepth=10, seed=SEED),
        "Gradient-Boosted Trees": GBTRegressor(featuresCol="features", labelCol="label", maxIter=80, maxDepth=6, seed=SEED),
    }
    fitted, preds = {}, {}
    for name, est in models.items():
        pm = timed(f"fit_{name}", T, lambda est=est: Pipeline(stages=[ohe, asm, est]).fit(train))
        p = pm.transform(test).cache()
        metrics[name] = score(p)
        fitted[name], preds[name] = pm, p
    R["forecast_metrics"] = metrics
    best_name = min((n for n in models), key=lambda n: metrics[n]["rmse"])
    R["best_model"] = best_name
    R["test_label_mean"] = round(test.agg(F.avg("label")).collect()[0][0], 2)

    # feature importance of the tree models
    names = ["hour", "dow", "is_weekend", "zone_lat", "zone_lon", "lag_24", "lag_168"] + [f"zone_{i}" for i in range(best_k)]
    imp = fitted["Gradient-Boosted Trees"].stages[-1].featureImportances.toArray()
    top = sorted(zip(names, imp), key=lambda x: -x[1])
    R["gbt_feature_importance"] = [{"feature": n, "importance": round(float(v), 4)} for n, v in top[:8]]
    zone_imp = sum(v for n, v in zip(names, imp) if n.startswith("zone_") and n not in ("zone_lat", "zone_lon"))
    R["gbt_zone_ohe_total_importance"] = round(float(zone_imp), 4)

    # actual vs predicted for the busiest zone in the last 14 days of the data
    top_zone = centers[0]["zone"]
    ap = (preds[best_name].filter(F.col("zone") == top_zone).filter(F.col("date") >= "2014-09-17")
          .select(F.concat_ws(" ", F.col("date").cast("string"), F.format_string("%02d:00", F.col("hour"))).alias("slot"),
                  "label", F.round("prediction", 1).alias("pred")).orderBy("slot").collect())
    write_csv(ap, f"{args.out}/forecast_actual_vs_pred.csv", ["slot", "label", "pred"])
    R["forecast_plot_zone"] = {"zone": top_zone, "landmark": centers[0]["landmark"]}

    # per-hotspot error of best model
    zerr = (preds[best_name].withColumn("err", F.abs(F.col("label") - F.col("prediction")))
            .groupBy("zone").agg(F.round(F.avg("err"), 2).alias("mae"), F.round(F.avg("label"), 2).alias("avg_demand")).collect())
    lm = {c["zone"]: c["landmark"] for c in centers}
    R["zone_error_best_model"] = sorted(
        [{"zone": r["zone"], "landmark": lm[r["zone"]], "mae": r["mae"], "avg_demand": r["avg_demand"]} for r in zerr],
        key=lambda d: -d["avg_demand"])

    # ------------------------------------------------------------------ 7. STORAGE BENCHMARK
    def bench_csv():
        d = (spark.read.csv(csv_files, header=True, schema=schema)
             .withColumn("h", F.hour(F.to_timestamp("ts_raw", "M/d/yyyy H:mm:ss"))))
        return d.groupBy("h").count().collect()

    def bench_parquet():
        return spark.read.parquet(pq_path).groupBy("hour").count().collect()

    def bench_parquet_pruned():
        return spark.read.parquet(pq_path).filter(F.col("month") == 9).groupBy("hour").count().collect()

    bench = {}
    for label, fn in [("CSV (all 6 months, parse + aggregate)", bench_csv),
                      ("Parquet (all 6 months)", bench_parquet),
                      ("Parquet (September only, partition pruning)", bench_parquet_pruned)]:
        runs = []
        for _ in range(3):
            t0 = time.time(); fn(); runs.append(time.time() - t0)
        bench[label] = {"runs_s": [round(x, 2) for x in runs], "median_s": round(statistics.median(runs), 2)}
        print(f"[bench] {label}: {bench[label]}", flush=True)
    R["storage_benchmark"] = bench

    # plan evidence for partition pruning
    try:
        plan = spark.read.parquet(pq_path).filter(F.col("month") == 9)._jdf.queryExecution().executedPlan().toString()
        R["partition_pruning_plan_has_filter"] = "PartitionFilters" in plan and "month" in plan
        R["partition_pruning_plan_excerpt"] = [ln.strip() for ln in plan.splitlines() if "PartitionFilters" in ln][:1]
    except Exception as e:  # plan introspection is informational only
        R["partition_pruning_plan_has_filter"] = None
        print("plan introspection skipped:", e)

    with open(os.path.join(args.out, "results.json"), "w") as fh:
        json.dump(R, fh, indent=2, default=str)
    print("DONE. Results written to", os.path.join(args.out, "results.json"))
    spark.stop()


if __name__ == "__main__":
    main()
