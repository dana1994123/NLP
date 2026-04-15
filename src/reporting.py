from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 


def summary_to_row(task, model, instances, summary):
    return {
        "Task": task,
        "Model": model,
        "Instances": instances,
        "Parsed_Count": summary.get("parsed_count"),
        "Valid_Count": summary.get("valid_count"),
        "Stable_Count": summary.get("stable_count"),
        "Exact_Match_Count": summary.get("exact_match_count"),
        "Correct_Count": summary.get("correct_count"),
        "Accuracy": summary.get("accuracy"),
        "Avg_Blocking_Pairs": summary.get("avg_blocking_pairs"),
    }


def build_final_summary_table(summary_map):
    rows = []

    for key, summary in summary_map.items():
        task, model, instances = key
        rows.append(summary_to_row(task, model, instances, summary))

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["Task", "Instances", "Model"]).reset_index(drop=True)
    return df


# def build_chart_dataframe(summary_map):
#     chart_rows = []

#     for (task, model, instances), summary in summary_map.items():
#         if task in ["Task 1", "Task 3"]:
#             score = summary.get("stable_count")
#         elif task in ["Task 2", "Task 4"]:
#             score = summary.get("accuracy")
#         else:
#             score = None

#         chart_rows.append({
#             "Task": task,
#             "Model": model,
#             "Instances": instances,
#             "Score": score
#         })

#     df = pd.DataFrame(chart_rows)
#     df = df.sort_values(by=["Task", "Instances", "Model"]).reset_index(drop=True)
#     return df


def build_chart_dataframe(summary_map):
    chart_rows = []

    for (task, model, instances), summary in summary_map.items():
        model = str(model).strip()

        if task == "Task 1":
            task_name = "Stable Matching\nGeneration"
            score = (
                summary.get("stable_count", 0) /
                summary.get("total_instances", 1)
            ) * 100

        elif task == "Task 2":
            task_name = "Instability\nDetection"
            score = summary.get("accuracy", 0) * 100

        elif task == "Task 3":
            task_name = "Instability\nResolution"
            score = (
                summary.get("stable_count", 0) /
                summary.get("total_instances", 1)
            ) * 100

        elif task == "Task 4":
            task_name = "Preference\nReasoning"
            score = summary.get("accuracy", 0) * 100

        else:
            continue

        chart_rows.append({
            "Task": task_name,
            "Model": model,
            "Score": score
        })

    df = pd.DataFrame(chart_rows)

    task_order = [
        "Stable Matching\nGeneration",
        "Instability\nDetection",
        "Instability\nResolution",
        "Preference\nReasoning"
    ]

    df["Task"] = pd.Categorical(df["Task"], categories=task_order, ordered=True)

    df = df.sort_values(by=["Task", "Model"]).reset_index(drop=True)

    return df
def plot_grouped_bar_chart(chart_df, save_path=None):
    pivot_df = chart_df.pivot(index="Task", columns="Model", values="Score")

    if "Basic" not in pivot_df.columns:
        pivot_df["Basic"] = 0
    if "Reasoning" not in pivot_df.columns:
        pivot_df["Reasoning"] = 0

    pivot_df = pivot_df[["Basic", "Reasoning"]]

    x = np.arange(len(pivot_df.index))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(
        x - width / 2,
        pivot_df["Basic"],
        width,
        label="Basic",
        edgecolor="black",
        linewidth=0.8
    )

    bars2 = ax.bar(
        x + width / 2,
        pivot_df["Reasoning"],
        width,
        label="Reasoning",
        edgecolor="black",
        linewidth=0.8
    )

    ax.set_title("Performance Comparison Across Tasks", fontsize=14, fontweight="bold")
    ax.set_ylabel("Performance (%)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Task", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_df.index, fontsize=10)
    ax.set_ylim(0, 100)

    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend()

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(
                f"{h:.1f}%",
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9
            )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
    return fig

# def plot_grouped_bar_chart(chart_df, save_path=None):
    pivot_df = chart_df.copy()
    pivot_df["Label"] = pivot_df["Task"] + " (" + pivot_df["Instances"].astype(str) + ")"

    plot_df = pivot_df.pivot(index="Label", columns="Model", values="Score")

    fig, ax = plt.subplots(figsize=(12, 6))
    plot_df.plot(kind="bar", ax=ax)

    ax.set_title("Basic vs Reasoning Performance Across Tasks")
    ax.set_ylabel("Score")
    ax.set_xlabel("Task and Instance Count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig