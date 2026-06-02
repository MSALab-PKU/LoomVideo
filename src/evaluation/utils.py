import os
import pandas as pd


def merge_and_save_temp_metrics(
    output_dir: str,
    final_csv_path: str,
    merge_columns: list,
    accelerator,
):
    """Merge per-rank temporary metric CSVs into a single final CSV.

    Each rank writes ``temp_metrics_{rank}.csv`` during evaluation.
    This function concatenates them and merges scores into the final CSV
    keyed by ``video_path``.

    Args:
        output_dir: Directory containing per-rank temp CSVs.
        final_csv_path: Path to the final merged CSV.
        merge_columns: List of column names to merge from temp into final.
        accelerator: HuggingFace Accelerator instance.
    """
    temp_dfs = []
    for proc_idx in range(accelerator.num_processes):
        temp_path = os.path.join(output_dir, f"temp_metrics_{proc_idx}.csv")
        if os.path.exists(temp_path):
            df = pd.read_csv(temp_path)
            temp_dfs.append(df)
            os.remove(temp_path)

    temp_combined = pd.concat(temp_dfs, ignore_index=True)

    if os.path.exists(final_csv_path):
        final_df = pd.read_csv(final_csv_path)
    else:
        final_df = temp_combined.copy()

    for _, temp_row in temp_combined.iterrows():
        video_path = temp_row["video_path"]
        mask = final_df["video_path"] == video_path
        if mask.any():
            for col in merge_columns:
                final_df.loc[mask, col] = temp_row[col]

    final_df.to_csv(final_csv_path, index=False, encoding="utf-8")
