import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
NAVY,BLUE,LIGHT,AMBER="#1F4E79","#4472C4","#DEEAF6","#ED7D31"
def box(ax,x,y,w,h,t,fc=LIGHT,ec=BLUE,tc="#1F2D3D",fs=9.5,bold=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.02,rounding_size=0.08",fc=fc,ec=ec,lw=1.6))
    ax.text(x+w/2,y+h/2,t,ha="center",va="center",fontsize=fs,color=tc,fontweight="bold" if bold else "normal",linespacing=1.3)
def arr(ax,a,b):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle="-|>",mutation_scale=14,color=NAVY,lw=1.6))
# workflow
fig,ax=plt.subplots(figsize=(10,2.6)); ax.set_xlim(0,10); ax.set_ylim(0,2.6); ax.axis("off")
steps=["Raw CSV\n(6 files,\n4.53M rows)","Ingest\nexplicit\nschema","Clean &\ntransform\n(PySpark)","Parquet\npartitioned\nby month","Spark SQL\nanalytics","K-Means\nhotspots\n(MLlib)","Demand\nforecast\n(MLlib)","Insights &\ncharts"]
w=1.0;g=0.2
for i,t in enumerate(steps):
    x=0.1+i*(w+g)*0.98
    box(ax,x,0.7,w,1.3,t,fc=NAVY if i in(0,7) else LIGHT,tc="white" if i in(0,7) else "#1F2D3D",ec=NAVY,fs=8.6)
    if i<7: arr(ax,(x+w+0.01,1.35),(x+w+g*0.98-0.01,1.35))
ax.text(5,0.3,"Stages 2-7 run in one PySpark pipeline; every stage after Parquet reads the cleaned storage layer",ha="center",fontsize=9,color="#555")
fig.savefig("out/figures/diagram_workflow.png",dpi=200,bbox_inches="tight",facecolor="white")
# architecture
fig,ax=plt.subplots(figsize=(10,4.6)); ax.set_xlim(0,10); ax.set_ylim(0,4.6); ax.axis("off")
cols=[("DATA SOURCE",0.1),("STORAGE LAYER",2.65),("PROCESSING ENGINE",5.0),("OUTPUT LAYER",7.9)]
for t,x in cols: ax.text(x+1.0,4.3,t,ha="center",fontsize=9.5,fontweight="bold",color=NAVY)
box(ax,0.1,2.4,2.0,1.4,"FiveThirtyEight\nGitHub repository\n6 monthly CSV files\n(209.7 MB)",fc=NAVY,tc="white",ec=NAVY)
box(ax,2.65,2.9,2.0,0.9,"Raw CSV\n(local disk)")
box(ax,2.65,1.3,2.0,0.9,"Parquet, partitioned\nby month (21.1 MB)",fc="#FCE4D6",ec=AMBER)
box(ax,5.0,3.15,2.45,0.7,"PySpark DataFrames:\ningest, clean, transform",fs=9)
box(ax,5.0,2.2,2.45,0.7,"Spark SQL analytics\n(hour, weekday, base)",fs=9)
box(ax,5.0,1.25,2.45,0.7,"MLlib K-Means\n(demand hotspots)",fs=9)
box(ax,5.0,0.3,2.45,0.7,"MLlib regression\n(day-ahead forecast)",fs=9)
box(ax,7.9,2.9,2.0,0.9,"Result tables\n(CSV / JSON)")
box(ax,7.9,1.5,2.0,0.9,"Charts and maps\n(matplotlib)")
box(ax,7.9,0.3,2.0,0.9,"Report and\npresentation",fc=NAVY,tc="white",ec=NAVY)
arr(ax,(2.1,3.1),(2.65,3.35)); arr(ax,(4.65,3.35),(5.0,3.5)); arr(ax,(4.65,3.1),(3.9,2.2)) if False else None
arr(ax,(6.2,3.15),(3.65,2.2)); arr(ax,(4.65,1.75),(5.0,2.5)); arr(ax,(4.65,1.6),(5.0,1.6)); arr(ax,(4.65,1.45),(5.0,0.7))
arr(ax,(7.45,2.5),(7.9,3.25)); arr(ax,(7.45,1.6),(7.9,1.9)); arr(ax,(7.45,0.65),(7.9,0.75)); arr(ax,(8.9,1.5),(8.9,1.2))
ax.text(5,0.0,"Spark runs in local mode on one machine (no cluster); the same code can be submitted to a cluster.",ha="center",fontsize=9,color="#555")
fig.savefig("out/figures/diagram_architecture.png",dpi=200,bbox_inches="tight",facecolor="white")
