import json
import numpy as np
import matplotlib.pyplot as plt


with open("evaluation_results.json") as f:
    data = json.load(f)

methods = ["baseline", "contextual", "full_contextual", "agentic"]

metrics = ["grounding", "personalization", "specificity", "actionability"]

metric_map = {
    "grounding": "Groundedness",
    "personalization": "Personalization",
    "specificity": "Specificity",
    "actionability": "Actionability"
}

heuristic = {m: {k: [] for k in metrics} for m in methods}
gpt = {m: {k: [] for k in metrics} for m in methods}


for entry in data:
    h = entry["heuristic"]
    g = entry["llm_judge"]

    for m in methods:
        for k in metrics:
            heuristic[m][k].append(h[m][k])
            gpt[m][k].append(g[m][metric_map[k]] / 2.0)  


def mean_ci(values):
    arr = np.array(values)
    mean = arr.mean()
    ci = 1.96 * arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else 0
    return mean, ci


heur_mean, heur_ci = {}, {}
gpt_mean, gpt_ci = {}, {}

for m in methods:
    heur_mean[m], heur_ci[m] = {}, {}
    gpt_mean[m], gpt_ci[m] = {}, {}

    for k in metrics:
        heur_mean[m][k], heur_ci[m][k] = mean_ci(heuristic[m][k])
        gpt_mean[m][k], gpt_ci[m][k] = mean_ci(gpt[m][k])

# overall performance plot
def plot_fig1_overall():
    x = np.arange(len(methods))
    width = 0.35

    heur_overall = [np.mean(list(heur_mean[m].values())) for m in methods]
    gpt_overall = [np.mean(list(gpt_mean[m].values())) for m in methods]

    heur_err = [np.mean(list(heur_ci[m].values())) for m in methods]
    gpt_err = [np.mean(list(gpt_ci[m].values())) for m in methods]

    plt.figure(figsize=(7, 4))

    plt.bar(x - width/2, heur_overall, width, yerr=heur_err,
            capsize=4, label="Heuristic", alpha=0.8)

    plt.bar(x + width/2, gpt_overall, width, yerr=gpt_err,
            capsize=4, label="GPT Judge", alpha=0.8)

    plt.xticks(x, methods, rotation=15)
    plt.ylabel("Score (0–1)")
    plt.title("Figure 1: Overall Performance Across Methods")
    plt.legend()

    plt.tight_layout()
    plt.savefig("fig1_overall.png", dpi=300)
    plt.close()


# metric-specific performance plot
def plot_fig2_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]

        heur_vals = [heur_mean[m][metric] for m in methods]
        gpt_vals = [gpt_mean[m][metric] for m in methods]

        ax.plot(methods, heur_vals, marker="o", label="Heuristic")
        ax.plot(methods, gpt_vals, marker="o", label="GPT")

        ax.set_title(metric.capitalize())
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=20)

    axes[0].legend()

    plt.tight_layout()
    plt.savefig("fig2_metrics.png", dpi=300)
    plt.close()

plot_fig1_overall()
plot_fig2_metrics()

print("\nSaved paper-quality figures:")
print("- fig1_overall.png")
print("- fig2_metrics.png")
