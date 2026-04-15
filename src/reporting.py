from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


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


def build_chart_dataframe(summary_map):
    chart_rows = []

    for (task, model, instances), summary in summary_map.items():
        if task in ["Task 1", "Task 3"]:
            score = summary.get("stable_count")
        elif task in ["Task 2", "Task 4"]:
            score = summary.get("accuracy")
        else:
            score = None

        chart_rows.append({
            "Task": task,
            "Model": model,
            "Instances": instances,
            "Score": score
        })

    df = pd.DataFrame(chart_rows)
    df = df.sort_values(by=["Task", "Instances", "Model"]).reset_index(drop=True)
    return df


def plot_grouped_bar_chart(chart_df, save_path=None):
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