import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Wedge

orig    = pd.read_csv("fragrantica_data/perfumes_table.csv")
filled1 = pd.read_csv("fragrantica_data/fragnantica_missing_filled.csv")
filled2 = pd.read_csv("fragrantica_data/missing_part1_filled.csv")
filled3 = pd.read_csv("fragrantica_data/missing_part2_filled.csv")
filled4 = pd.read_csv("fragrantica_data/missing_part3_filled.csv")

filled = pd.concat([filled1, filled2, filled3, filled4], ignore_index=True)
filled = filled.drop_duplicates(subset="url", keep="first")

total           = len(orig)
orig_has_rating = int(orig["rating"].notna().sum())
total_missing   = int(orig["rating"].isna().sum())
scraped         = len(filled)
scraped_filled  = int(filled["rating"].notna().sum())
not_found       = int((filled["parfumo_url"] == "NOT_FOUND").sum())
scraped_empty   = scraped - scraped_filled - not_found
not_yet         = total_missing - scraped

# styling
BG      = "#FAFAF7"
PANEL   = "#FFFFFF"
INK     = "#1A1A2E"
MUTED   = "#8B8B9E"
FAINT   = "#B8B8C8"
GRID    = "#EEEEF2"

C = {
    "orig":     "#3D8B6B",
    "filled":   "#4A6FA5",
    "notfound": "#C9534F",
    "empty":    "#D9925A",
    "notyet":   "#DCDCE4",
}

plt.rcParams.update({
    "font.family":       ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "text.color":        INK,
    "axes.labelcolor":   INK,
    "xtick.color":       MUTED,
    "ytick.color":       MUTED,
    "axes.edgecolor":    GRID,
    "axes.linewidth":    0.8,
    "axes.facecolor":    PANEL,
    "figure.facecolor":  BG,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        GRID,
    "grid.linewidth":    0.7,
    "xtick.major.size":  0,
    "ytick.major.size":  0,
    "xtick.major.pad":   6,
    "ytick.major.pad":   6,
})

fig = plt.figure(figsize=(15, 11))
fig.patch.set_facecolor(BG)

# header  
fig.text(0.06, 0.965, "F R A G R A N T I C A",
         ha="left", va="top", fontsize=9, color=MUTED)
fig.text(0.06, 0.94, "Vul-voortgang",
         ha="left", va="top", fontsize=28, fontweight="bold", color=INK)
fig.text(0.06, 0.905,
         f"Status van {total:,} parfums uit de Fragrantica-dataset, "
         f"verrijkt met ratings van Parfumo.",
         ha="left", va="top", fontsize=10.5, color=MUTED)

fig.text(0.97, 0.94, f"{total:,}", ha="right", va="top",
         fontsize=30, fontweight="bold", color=INK)
fig.text(0.97, 0.905, "PARFUMS TOTAAL", ha="right", va="top",
         fontsize=8.5, color=MUTED)

fig.add_artist(plt.Line2D([0.06, 0.97], [0.88, 0.88],
                          color=GRID, linewidth=1, transform=fig.transFigure))

# layout  
gs = fig.add_gridspec(2, 2, top=0.78, bottom=0.06,
                      left=0.06, right=0.97,
                      hspace=0.75, wspace=0.20,
                      height_ratios=[1, 2.0])
ax_bar     = fig.add_subplot(gs[0, :])
ax_prog    = fig.add_subplot(gs[1, 0])
ax_scraped = fig.add_subplot(gs[1, 1])


def panel_header(ax, num, title, subtitle=None):
    ax.text(0, 1.22, f"0{num}    {title.upper()}",
            transform=ax.transAxes,
            fontsize=9, color=MUTED, fontweight="bold")
    if subtitle:
        ax.text(0, 1.10, subtitle, transform=ax.transAxes,
                fontsize=12, color=INK)


# Stacked bar
segs = [
    (orig_has_rating, C["orig"],     "Al rating in origineel"),
    (scraped_filled,  C["filled"],   "Gevuld via Parfumo"),
    (not_found,       C["notfound"], "Niet gevonden"),
    (scraped_empty,   C["empty"],    "Gescraped, geen rating"),
    (not_yet,         C["notyet"],   "Nog niet gescraped"),
]

bar_h = 0.45
gap   = max(total * 0.0012, 1)
left  = 0
handles = []
for val, color, label in segs:
    w = max(val - gap, 0)
    ax_bar.barh(0, w, left=left, color=color,
                edgecolor="none", height=bar_h, zorder=3)
    pct = val / total * 100
    if pct > 2.5:
        text_color = "white" if color != C["notyet"] else MUTED
        ax_bar.text(left + w / 2, 0, f"{pct:.1f}%",
                    ha="center", va="center", fontsize=10,
                    color=text_color, fontweight="bold", zorder=4)
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=color, ec="none",
                                 label=f"  {label}  ·  {val:,}"))
    left += val

ax_bar.set_xlim(-total*0.005, total*1.005)
ax_bar.set_ylim(-0.9, 0.9)
ax_bar.set_yticks([])
ax_bar.xaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x else "0"))
ax_bar.spines["left"].set_visible(False)
ax_bar.spines["bottom"].set_color(GRID)
ax_bar.grid(axis="y", visible=False)
ax_bar.grid(axis="x", alpha=0.5)
ax_bar.tick_params(labelsize=9)

panel_header(ax_bar, 1, "Overzicht", "Verdeling van alle parfums in de dataset")

leg = ax_bar.legend(
    handles=handles, loc="upper center",
    bbox_to_anchor=(0.5, -0.45), ncol=5,
    fontsize=9.5, frameon=False,
    handlelength=1.2, handleheight=1.0,
    columnspacing=1.8, handletextpad=0.5,
)
for text in leg.get_texts():
    text.set_color(INK)


# Donut progress 
ax_prog.set_aspect('equal')
ax_prog.set_xlim(-1.5, 1.5)
ax_prog.set_ylim(-1.6, 1.5)
ax_prog.axis('off')

pct_done = scraped / total_missing * 100 if total_missing else 0

ring_outer = 1.0
ring_inner = 0.74

ax_prog.add_patch(Wedge((0, 0), ring_outer, 0, 360,
                        width=ring_outer - ring_inner,
                        facecolor=C["notyet"], edgecolor="none", zorder=2))
if pct_done > 0:
    angle = 90 - (pct_done / 100) * 360
    ax_prog.add_patch(Wedge((0, 0), ring_outer, angle, 90,
                            width=ring_outer - ring_inner,
                            facecolor=C["filled"], edgecolor="none", zorder=3))

ax_prog.text(0, 0.12, f"{pct_done:.1f}%",
             ha="center", va="center",
             fontsize=34, fontweight="bold", color=INK)
ax_prog.text(0, -0.18, "voltooid",
             ha="center", va="center",
             fontsize=10, color=MUTED)

ax_prog.text(0, -1.28,
             f"{scraped:,} van {total_missing:,} ontbrekende parfums",
             ha="center", va="center",
             fontsize=10, color=INK, fontweight="bold")
ax_prog.text(0, -1.45,
             f"nog {not_yet:,} te gaan",
             ha="center", va="center",
             fontsize=9, color=MUTED)

panel_header(ax_prog, 2, "Voortgang", "Scraping van ontbrekende parfums")


# Breakdown 
bvals   = [scraped_filled, not_found, scraped_empty]
bcolors = [C["filled"], C["notfound"], C["empty"]]
bnames  = ["Rating\ngevonden", "Niet\ngevonden", "Geen\nrating"]

x_pos = list(range(len(bvals)))
bar_w = 0.5

ax_scraped.bar(x_pos, bvals, color=bcolors, width=bar_w,
               edgecolor="none", zorder=3)

y_max = max(bvals)
for i, v in enumerate(bvals):
    pct = v / scraped * 100 if scraped else 0
    ax_scraped.text(i, v + y_max*0.06,
                    f"{v:,}",
                    ha="center", va="bottom", fontsize=15,
                    fontweight="bold", color=INK)
    ax_scraped.text(i, v + y_max*0.015,
                    f"{pct:.1f}%",
                    ha="center", va="bottom", fontsize=9.5,
                    color=MUTED)

ax_scraped.set_xticks(x_pos)
ax_scraped.set_xticklabels(bnames, fontsize=10, color=INK)
ax_scraped.set_xlim(-0.7, len(bvals) - 0.3)
ax_scraped.set_ylim(0, y_max * 1.30)
ax_scraped.yaxis.set_major_formatter(mticker.FuncFormatter(
    lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
ax_scraped.spines["bottom"].set_color(GRID)
ax_scraped.grid(axis="x", visible=False)
ax_scraped.grid(axis="y", alpha=0.5)
ax_scraped.tick_params(axis="x", length=0)
ax_scraped.tick_params(labelsize=9)

panel_header(ax_scraped, 3, "Breakdown",
             f"Uitkomst van {scraped:,} scrape-pogingen")


# footer
fig.text(0.06, 0.025, "fragrantica × parfumo",
         fontsize=8.5, color=FAINT)
fig.text(0.97, 0.025, "matplotlib",
         fontsize=8.5, color=FAINT, ha="right")

plt.savefig("fragrantica_fill_progress.png", dpi=180, bbox_inches="tight",
            facecolor=BG)
plt.show()
print("Opgeslagen als fragrantica_fill_progress.png")