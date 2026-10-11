import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")

def plot(ycol, ylabel, title, fname, ideal=False):
    plt.figure(figsize=(7, 4.5))
    for n, g in df.groupby("n"):
        plt.plot(g["threads"], g[ycol], marker="o", label=f"N = {n:,}")
    if ideal:
        ts = sorted(df["threads"].unique())
        plt.plot(ts, ts, "k--", label="Ideal")
    plt.xlabel("Number of threads"); plt.ylabel(ylabel); plt.title(title)
    plt.xticks(sorted(df["threads"].unique())); plt.grid(True); plt.legend()
    plt.tight_layout(); plt.savefig(f"graphs/{fname}", dpi=150); plt.close()

plot("par_time", "Execution time (s)", "Execution Time vs Threads", "execution_time.png")
plot("speedup", "Speedup (Ts / Tp)", "Speedup vs Threads", "speedup.png", ideal=True)
plot("efficiency", "Efficiency (Speedup / p)", "Efficiency vs Threads", "efficiency.png")
print("Graphs saved in graphs/")
