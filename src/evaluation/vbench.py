"""
VBench evaluation: imaging quality, overall consistency, subject consistency.
"""

import os
import json
from collections import defaultdict

import pandas as pd
import torch
from pyiqa.archs.musiq_arch import MUSIQ

from libs.VBench.vbench.utils import CACHE_DIR
from libs.VBench.vbench.third_party.ViCLIP.viclip import ViCLIP
from libs.VBench.vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer

from libs.VBench.vbench.utils import init_submodules
from libs.VBench.vbench.imaging_quality import technical_quality
from libs.VBench.vbench.overall_consistency import overall_consistency
from libs.VBench.vbench.subject_consistency import subject_consistency

DIMENSION_LIST = ["imaging_quality", "overall_consistency", "subject_consistency"]


def evaluate_on_vbench(
    dataset,
    eval_root: str,
    accelerator,
    config,
):
    """
    Evaluate generated videos on VBench dimensions.

    Loads pretrained models for each dimension once, then distributes
    evaluation across ranks. Per-rank results are written as temporary CSVs,
    then merged by the main process into ``vbench_each.csv`` and ``vbench.json``.

    Args:
        dataset: UniEvalDataset instance.
        eval_root: Root directory containing generated outputs.
        accelerator: HuggingFace Accelerator instance.
        config: Full config (unused but kept for API consistency).
    """
    score_columns = [f"vbench_{dim}_score" for dim in DIMENSION_LIST]

    # Load pretrained models
    submodules_dict = init_submodules(DIMENSION_LIST)

    if "imaging_quality" in DIMENSION_LIST:
        submodules_list = submodules_dict["imaging_quality"]
        model_path = submodules_list["model_path"]
        iq_model = MUSIQ(pretrained_model_path=model_path)
        iq_model.to(accelerator.device)
        iq_model.training = False

    if "overall_consistency" in DIMENSION_LIST:
        submodules_list = submodules_dict["overall_consistency"]
        tokenizer = SimpleTokenizer(
            os.path.join(CACHE_DIR, "ViCLIP/bpe_simple_vocab_16e6.txt.gz")
        )
        viclip = ViCLIP(tokenizer=tokenizer, **submodules_list).to(accelerator.device)

    if "subject_consistency" in DIMENSION_LIST:
        submodules_list = submodules_dict["subject_consistency"]
        dino_model = torch.hub.load(**submodules_list).to(accelerator.device)
        sc_read_frame = submodules_list["read_frame"]

    accelerator.wait_for_everyone()

    # Each rank collects rows grouped by dataset_name
    temp_rows = defaultdict(list)

    with accelerator.split_between_processes(list(range(len(dataset)))) as local_indices:
        for idx in local_indices:
            data = dataset[idx]
            if data is None:
                continue

            dataset_name = data["dataset_name"]
            instruction = data["instruction"]

            # Per-sample dimension filtering
            sample_dims = data.get("data_info", {}).get("dimensions", DIMENSION_LIST)

            for gen_relative_path in data["gen_paths"]:
                video_path = os.path.join(
                    eval_root, dataset_name, "outputs", gen_relative_path
                )

                # Skip images
                if video_path.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    continue

                if not os.path.exists(video_path):
                    continue

                scores = {}

                if "imaging_quality" in sample_dims:
                    _, video_results = technical_quality(
                        iq_model, [video_path], accelerator.device
                    )
                    scores["vbench_imaging_quality_score"] = (
                        video_results[0]["video_results"] / 100.0
                    )

                if "overall_consistency" in sample_dims:
                    video_dict = [{"prompt": instruction, "video_list": [video_path]}]
                    _, video_results = overall_consistency(
                        viclip, video_dict, tokenizer, accelerator.device
                    )
                    scores["vbench_overall_consistency_score"] = video_results[0][
                        "video_results"
                    ]

                if "subject_consistency" in sample_dims:
                    _, video_results = subject_consistency(
                        dino_model, [video_path], accelerator.device, sc_read_frame
                    )
                    scores["vbench_subject_consistency_score"] = video_results[0][
                        "video_results"
                    ]

                row = {"gen_path": video_path, "instruction": instruction}
                for col in score_columns:
                    row[col] = scores.get(col, float("nan"))
                temp_rows[dataset_name].append(row)

    # Write per-rank temp CSVs
    for dataset_name, rows in temp_rows.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        temp_path = os.path.join(
            metrics_dir, f"vbench_each_rank{accelerator.process_index}.csv"
        )
        pd.DataFrame(rows).to_csv(temp_path, index=False)

    accelerator.wait_for_everyone()

    # Main process: merge temp CSVs into vbench_each.csv and vbench.json
    if accelerator.is_main_process:
        for dataset_name in dataset.dataset_names:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            temp_dfs = []
            for proc_idx in range(accelerator.num_processes):
                temp_path = os.path.join(
                    metrics_dir, f"vbench_each_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    temp_dfs.append(pd.read_csv(temp_path))
                    os.remove(temp_path)

            if not temp_dfs:
                continue

            merged_df = pd.concat(temp_dfs, ignore_index=True)
            merged_df.to_csv(
                os.path.join(metrics_dir, "vbench_each.csv"), index=False
            )

            averages = {
                col: float(merged_df[col].dropna().mean())
                for col in score_columns
                if col in merged_df.columns
            }
            with open(
                os.path.join(metrics_dir, "vbench.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(averages, f, indent=2)

    accelerator.wait_for_everyone()
