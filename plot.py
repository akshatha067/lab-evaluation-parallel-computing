import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/results.csv")
seq = df[df.method == "seq"].set_index("gb")["time"]
par = df[df.method != "seq"].copy()
par["speedup"] = par.apply(lambda r: seq[r.gb] / r.time, axis=1)
par["efficiency"] = par.speedup / par.workers
par.to_csv("results/summary.csv", index=False)
print(par[["method", "gb", "workers", "time", "speedup", "efficiency"]].round(3).to_string(index=False))

colors = {"pthreads": "tab:blue", "mpi": "tab:red"}
for gb, g in par.groupby("gb"):
    plt.figure(figsize=(7, 4.5))
    plt.axhline(seq[gb], color="k", linestyle="--", label=f"Sequential ({seq[gb]:.1f} s)")
    for m, d in g.groupby("method"):
        plt.plot(d.workers, d.time, marker="o", color=colors[m], label=m)
    plt.xlabel("Threads / MPI processes"); plt.ylabel("Execution time (s)")
    plt.title(f"Execution time, {gb:g} GB dataset"); plt.xscale("log", base=2)
    plt.xticks(sorted(g.workers.unique()), sorted(g.workers.unique()))
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(f"graphs/time_{gb:g}GB.png", dpi=150); plt.close()

for col, ylabel, fname, ideal in [("speedup", "Speedup (Tseq / Tpar)", "speedup.png", True),
                                  ("efficiency", "Efficiency (speedup / workers)", "efficiency.png", False)]:
    plt.figure(figsize=(7, 4.5))
    for (m, gb), d in par.groupby(["method", "gb"]):
        plt.plot(d.workers, d[col], marker="o", linestyle="-" if m == "pthreads" else "--",
                 label=f"{m} {gb:g} GB")
    if ideal:
        ws = sorted(par.workers.unique()); plt.plot(ws, ws, "k:", label="Ideal")
    plt.xlabel("Threads / MPI processes"); plt.ylabel(ylabel); plt.xscale("log", base=2)
    plt.xticks(sorted(par.workers.unique()), sorted(par.workers.unique()))
    plt.grid(True); plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(f"graphs/{fname}", dpi=150); plt.close()
print("Graphs saved in graphs/")
