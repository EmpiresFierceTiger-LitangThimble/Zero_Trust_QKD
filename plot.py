import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt
import datetime
import os


def plot_AS_vs_AP_from_npz(filename,
                           show_std=True,
                           out_dir="plots",
                           dpi=300):

    data = np.load(filename)
    num_nodes = data["num_nodes"]
    ap_list = data["ap_list"]
    mp_range = data["mp_range"]
    ASR_mean = data["ASR_mean"]
    ASR_std = data["ASR_std"]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(8, 5))

    for i, mp in enumerate(mp_range):
        if show_std:
            plt.errorbar(
                ap_list, ASR_mean[i],
                yerr=ASR_std[i],
                fmt="-o",
                capsize=4,
                label=f"num_multipath = {mp}"
            )
        else:
            plt.plot(ap_list, ASR_mean[i], "-o", label=f"num_multipath = {mp}")

    plt.xlabel("Attack Pervasiveness (AP), a.u.")
    plt.ylabel("Attack Success Rate (ASR), a.u.")
    plt.title(f"ASR vs AP, {num_nodes}")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(out_dir, f"{num_nodes}_AS_vs_AP_{timestamp}.png")

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Figure saved to {save_path}")

def plot_traditional_multi_mp_from_npz(filename,
                                       show_std=True,
                                       out_dir="plots",
                                       dpi=300):

    data = np.load(filename, allow_pickle=True)
    num_nodes = int(data["num_nodes"])
    results = data["results"].item()   # results 是一个字典，需要 .item()

    mp_list = sorted(results.keys())

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.figure(figsize=(8, 5))

    for mp in mp_list:
        ap_list = results[mp]["ap_list"]
        ASR_mean = results[mp]["ASR_mean"]
        ASR_std  = results[mp]["ASR_std"]

        if show_std:
            plt.errorbar(
                ap_list, ASR_mean,
                yerr=ASR_std,
                fmt="-o",
                capsize=4,
                label=f"num_multipath = {mp}"
            )
        else:
            plt.plot(ap_list, ASR_mean, "-o", label=f"num_multipath = {mp}")

    plt.xlabel("Attack Pervasiveness (AP), a.u.")
    plt.ylabel("Attack Success Rate (ASR), a.u.")
    plt.title(f"Traditional Multipath ASR vs AP, n={num_nodes}")
    plt.grid(True)
    plt.legend()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_name = f"traditional_multi_mp_n={num_nodes}_{timestamp}.png"
    save_path = os.path.join(out_dir, out_name)

    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    print(f"[Saved] Figure saved to {save_path}")


if __name__ == "__main__":
    plot_traditional_multi_mp_from_npz(
        "raw_data/traditional_multi_mp_n=100_2025-12-04_16-57-39.npz",
        show_std=True,
        out_dir="plots",
        dpi=300
    )
