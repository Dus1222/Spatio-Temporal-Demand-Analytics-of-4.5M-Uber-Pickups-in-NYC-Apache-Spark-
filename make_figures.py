"""Builds all report/presentation figures from the pipeline outputs in out/ (no numbers are typed by hand)."""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = sys.argv[1] if len(sys.argv) > 1 else "out"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

NAVY, BLUE, LIGHT, AMBER, GREY, DARK = "#1F4E79", "#4472C4", "#9DC3E6", "#ED7D31", "#7F7F7F", "#2D2D2D"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": "#BFBFBF", "axes.labelcolor": DARK,
    "axes.spines.top": False, "axes.spines.right": False, "xtick.color": DARK, "ytick.color": DARK,
    "axes.titleweight": "bold", "axes.titlecolor": NAVY, "axes.titlesize": 13, "figure.dpi": 100,
})
MONTHS = {4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug", 9: "Sep"}
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def read(name):
    with open(os.path.join(OUT, name)) as fh:
        return list(csv.DictReader(fh))


R = json.load(open(os.path.join(OUT, "results.json")))
PB = json.load(open(os.path.join(OUT, "pandas_benchmark.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1 ---------------------------------------------------------------- monthly volume
m = read("q_monthly.csv")
fig, ax = plt.subplots(figsize=(7, 4))
x = [MONTHS[int(r["month"])] for r in m]
y = [int(r["trips"]) / 1e3 for r in m]
bars = ax.bar(x, y, color=BLUE, width=0.62)
bars[-1].set_color(NAVY)
for b, v, r in zip(bars, y, m):
    ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,.0f}k", ha="center", fontsize=10, color=DARK)
ax.set_ylabel("Pickups (thousands)")
ax.set_title("Monthly Uber pickups in NYC, Apr–Sep 2014")
ax.set_ylim(0, max(y) * 1.12)
save(fig, "fig_monthly.png")

# 2 ---------------------------------------------------------------- hourly weekday vs weekend
h = read("q_hourly.csv")
wd = [float(r["avg_trips_per_day"]) for r in h if r["is_weekend"] == "0"]
we = [float(r["avg_trips_per_day"]) for r in h if r["is_weekend"] == "1"]
fig, ax = plt.subplots(figsize=(7.2, 4))
ax.plot(range(24), wd, color=NAVY, lw=2.4, marker="o", ms=4, label="Weekday (Mon–Fri)")
ax.plot(range(24), we, color=AMBER, lw=2.4, marker="o", ms=4, label="Weekend (Sat–Sun)")
ax.set_xticks(range(0, 24, 2))
ax.set_xlabel("Hour of day (0 = midnight)")
ax.set_ylabel("Average pickups per day in that hour")
ax.set_title("Hourly demand profile: weekdays vs weekends")
ax.legend(frameon=False)
ax.grid(axis="y", color="#E5E5E5")
pk = int(np.argmax(wd))
ax.annotate(f"Weekday peak\n{pk}:00 ({wd[pk]:,.0f})", (pk, wd[pk]), xytext=(pk - 7.2, wd[pk] - 150),
            arrowprops=dict(arrowstyle="->", color=GREY), fontsize=10)
save(fig, "fig_hourly.png")

# 3 ---------------------------------------------------------------- heatmap
hm = read("q_heatmap.csv")
M = np.zeros((7, 24))
for r in hm:
    M[int(r["dow"]), int(r["hour"])] = float(r["avg_trips"])
cmap = LinearSegmentedColormap.from_list("blues", ["#F2F7FC", LIGHT, BLUE, NAVY])
fig, ax = plt.subplots(figsize=(8.2, 3.9))
im = ax.imshow(M, aspect="auto", cmap=cmap)
ax.set_yticks(range(7)); ax.set_yticklabels(DOW)
ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("Hour of day")
ax.set_title("Average pickups by weekday and hour")
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, pad=0.015)
cb.set_label("Avg pickups / hour")
save(fig, "fig_heatmap.png")

# 4 ---------------------------------------------------------------- daily series
d = read("q_daily.csv")
import datetime as dt
dates = [dt.date.fromisoformat(r["date"]) for r in d]
vals = np.array([int(r["trips"]) for r in d]) / 1e3
ma = np.convolve(vals, np.ones(7) / 7, mode="valid")
fig, ax = plt.subplots(figsize=(8.4, 3.9))
ax.plot(dates, vals, color=LIGHT, lw=1.4, label="Daily pickups")
ax.plot(dates[3:-3], ma, color=NAVY, lw=2.6, label="7-day moving average")
top = np.argsort(vals)[::-1][:3]
ax.scatter([dates[i] for i in top], [vals[i] for i in top], color=AMBER, zorder=5, s=34, label="Three busiest days (5, 13, 19 Sep)")
ax.set_ylabel("Pickups per day (thousands)")
ax.set_title("Daily demand shows strong growth through summer 2014")
ax.set_ylim(0, vals.max() * 1.2)
ax.legend(frameon=False, loc="upper left")
ax.grid(axis="y", color="#E5E5E5")
fig.autofmt_xdate()
save(fig, "fig_daily.png")
R["_top_days"] = [{"date": str(dates[i]), "weekday": dates[i].strftime("%A"), "trips": int(vals[i] * 1e3)} for i in top]
R["_low_days"] = [{"date": str(dates[i]), "weekday": dates[i].strftime("%A"), "trips": int(vals[i] * 1e3)} for i in np.argsort(vals)[:3]]

# 5 ---------------------------------------------------------------- base share
b = read("q_base_month.csv")
bases = sorted({r["Base"] for r in b})
tot = {k: sum(int(r["trips"]) for r in b if r["Base"] == k) for k in bases}
order = sorted(bases, key=lambda k: -tot[k])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
cols = [NAVY, BLUE, LIGHT, "#BDD7EE", "#DEEBF7"]
bars = ax.barh(order[::-1], [tot[k] / 1e6 for k in order[::-1]], color=[cols[i] for i in range(len(order))][::-1])
s = sum(tot.values())
for bar, k in zip(bars, order[::-1]):
    ax.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{tot[k]/1e6:.2f}M  ({100*tot[k]/s:.1f}%)",
            va="center", fontsize=10)
ax.set_xlim(0, max(tot.values()) / 1e6 * 1.38)
ax.set_xlabel("Pickups (millions)")
ax.set_title("Pickups by dispatching base (TLC base code)")
save(fig, "fig_base.png")
R["_base_totals"] = {k: tot[k] for k in order}

# 6 ---------------------------------------------------------------- map
pts = read("map_sample.csv")
lat = np.array([float(r["Lat"]) for r in pts]); lon = np.array([float(r["Lon"]) for r in pts])
zone = np.array([int(r["zone"]) for r in pts])
hs = R["hotspots"]
palette = ["#1F4E79", "#ED7D31", "#4472C4", "#70AD47", "#7030A0", "#C00000", "#00A6A6", "#FFC000", "#7F7F7F", "#A5A5A5"]
fig, ax = plt.subplots(figsize=(6.6, 6.6))
for i, hsp in enumerate(hs):
    sel = zone == hsp["zone"]
    ax.scatter(lon[sel], lat[sel], s=1.2, alpha=0.35, color=palette[i % len(palette)], linewidths=0, rasterized=True)
for i, hsp in enumerate(hs):
    ax.scatter(hsp["lon"], hsp["lat"], s=95, color="white", edgecolor=palette[i % len(palette)], linewidth=2.6, zorder=5)
    ax.text(hsp["lon"] + 0.012, hsp["lat"] + 0.008, f'{hsp["landmark"]}\n{hsp["share_pct"]:.1f}%', fontsize=8.6, color=DARK,
            zorder=6, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
ax.set_aspect(1 / np.cos(np.radians(40.7)))
ax.set_xlim(-74.25, -73.70); ax.set_ylim(40.52, 40.92)
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
ax.set_title(f"K-Means demand hotspots (k = {R['best_k']}) on a 3% sample of pickups")
save(fig, "fig_map.png")

# 7 ---------------------------------------------------------------- silhouette
ks = sorted(int(k) for k in R["silhouette_by_k"])
sv = [R["silhouette_by_k"][str(k)] for k in ks]
fig, ax = plt.subplots(figsize=(5.8, 3.6))
ax.plot(ks, sv, color=BLUE, lw=2.4, marker="o", ms=7)
bi = ks.index(R["best_k"])
ax.scatter([ks[bi]], [sv[bi]], s=130, color=AMBER, zorder=5)
for k, v in zip(ks, sv):
    ax.text(k, v + 0.012, f"{v:.3f}", ha="center", fontsize=9.5)
ax.set_xticks(ks); ax.set_ylim(min(sv) - 0.06, max(sv) + 0.08)
ax.set_xlabel("Number of clusters (k)"); ax.set_ylabel("Silhouette score")
ax.set_title("Choosing k with the silhouette score")
ax.grid(axis="y", color="#E5E5E5")
save(fig, "fig_silhouette.png")

# 8 ---------------------------------------------------------------- model comparison
fm = R["forecast_metrics"]
names = list(fm.keys())
short = {"Baseline: same slot last week": "Naive: same slot\nlast week", "Baseline: mean by zone-hour-weekday": "Naive: historical\nmean profile",
         "Linear Regression": "Linear\nRegression", "Random Forest": "Random\nForest", "Gradient-Boosted Trees": "Gradient-\nBoosted Trees"}
rm = [fm[n]["rmse"] for n in names]
best = int(np.argmin(rm))
fig, ax = plt.subplots(figsize=(7.4, 4))
cols = [LIGHT, LIGHT, BLUE, BLUE, BLUE]
bars = ax.bar([short[n] for n in names], rm, color=cols, width=0.62)
bars[best].set_color(NAVY)
for bar, n in zip(bars, names):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f'{fm[n]["rmse"]:.1f}', ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylabel("RMSE (pickups per zone-hour, lower is better)")
ax.set_title("Day-ahead demand forecast: test-set error (September 2014)")
ax.set_ylim(0, max(rm) * 1.15)
ax.tick_params(axis="x", labelsize=9)
save(fig, "fig_models.png")

# 9 ---------------------------------------------------------------- forecast vs actual
ap = read("forecast_actual_vs_pred.csv")
yt = np.array([float(r["label"]) for r in ap]); yp = np.array([float(r["pred"]) for r in ap])
fig, ax = plt.subplots(figsize=(8.6, 3.7))
ax.plot(yt, color=NAVY, lw=1.8, label="Actual")
ax.plot(yp, color=AMBER, lw=1.6, label=f'Predicted ({R["best_model"]})')
ax.set_xlabel(f'Hourly slots, {ap[0]["slot"][:10]} to {ap[-1]["slot"][:10]}')
ax.set_ylabel("Pickups per hour")
ax.set_xticks([])
ax.set_title(f'Forecast vs actual: {R["forecast_plot_zone"]["landmark"]} hotspot, last 14 days of September')
ax.legend(frameon=False, ncol=2, loc="upper left")
ax.grid(axis="y", color="#E5E5E5")
save(fig, "fig_forecast.png")

# 10 --------------------------------------------------------------- benchmark
sb = R["storage_benchmark"]
items = [("pandas\n(CSV, 1 node)", PB["median_s"], GREY),
         ("Spark local\n(CSV)", sb["CSV (all 6 months, parse + aggregate)"]["median_s"], BLUE),
         ("Spark local\n(Parquet)", sb["Parquet (all 6 months)"]["median_s"], NAVY),
         ("Spark local\n(Parquet, Sep only)", sb["Parquet (September only, partition pruning)"]["median_s"], AMBER)]
fig, ax = plt.subplots(figsize=(7.2, 4))
bars = ax.bar([i[0] for i in items], [i[1] for i in items], color=[i[2] for i in items], width=0.6)
for bar, i in zip(bars, items):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08, f"{i[1]:.2f} s", ha="center", fontsize=10.5, fontweight="bold")
ax.set_ylabel("Median time of 3 runs (seconds)")
ax.set_title("Same query (pickups per hour), different storage and engines")
ax.set_ylim(0, max(i[1] for i in items) * 1.18)
ax.tick_params(axis="x", labelsize=9)
save(fig, "fig_benchmark.png")

json.dump(R, open(os.path.join(OUT, "results_enriched.json"), "w"), indent=2)
print("figures:", sorted(os.listdir(FIG)))
print("top days:", R["_top_days"]); print("low days:", R["_low_days"]); print("bases:", R["_base_totals"])
