"""
Metric dispatch module for evaluation.
"""

import os
import torch
from omegaconf import OmegaConf

from src.dataset.dataset import UniEvalDataset
from src.evaluation.vbench import evaluate_on_vbench
from src.evaluation.geneval import evaluate_on_geneval
from src.evaluation.openve import evaluate_on_openve
from src.evaluation.refvie import evaluate_on_refvie
from src.evaluation.imgedit import evaluate_on_imgedit
from src.evaluation.intelligent_vbench import evaluate_on_intelligent_vbench


@torch.no_grad()
def calculate_metrics(
    config,
    dataset_config,
    eval_root: str,
    accelerator,
    logger,
):
    """
    Run evaluation metrics on generated outputs.

    For each metric type requested by the dataset config, builds a UniEvalDataset
    containing the relevant sub-datasets and dispatches to the appropriate
    evaluation function.

    Args:
        config: Full training/evaluation config.
        dataset_config: OmegaConf dict with an ``eval`` key mapping dataset names
            to their configs (each must have a ``metrics`` list).
        eval_root: Root directory containing generated outputs.
        accelerator: HuggingFace Accelerator instance.
        logger: Logger instance.
    """
    # Build metric -> [dataset_name] mapping
    metric_to_datasets = {}
    for dataset_name, dataset_info in dataset_config.eval.items():
        for metric in dataset_info.metrics:
            if metric not in metric_to_datasets:
                metric_to_datasets[metric] = []
            metric_to_datasets[metric].append(dataset_name)

    # Pre-create metrics directories
    if accelerator.is_main_process:
        for dataset_name in dataset_config.eval:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Evaluate each metric
    for metric, dataset_names in metric_to_datasets.items():
        sub_config = OmegaConf.create(
            {name: dataset_config.eval[name] for name in dataset_names}
        )
        eval_dataset = UniEvalDataset(
            dataset_config=sub_config,
            config=config,
            gen_output_root=eval_root,
        )

        if metric == "vbench":
            logger.info("Start VBench evaluation!")
            evaluate_on_vbench(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish VBench evaluation!")

        elif metric == "geneval":
            logger.info("Start GenEval evaluation!")
            evaluate_on_geneval(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish GenEval evaluation!")

        elif metric == "openve":
            logger.info("Start OpenVE-Bench evaluation!")
            evaluate_on_openve(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish OpenVE-Bench evaluation!")

        elif metric == "refvie":
            logger.info("Start RefVIE-Bench evaluation!")
            evaluate_on_refvie(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish RefVIE-Bench evaluation!")

        elif metric == "imgedit":
            logger.info("Start ImgEdit-Bench evaluation!")
            evaluate_on_imgedit(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish ImgEdit-Bench evaluation!")

        elif metric == "intelligent_vbench":
            logger.info("Start Intelligent-VBench evaluation!")
            evaluate_on_intelligent_vbench(
                dataset=eval_dataset,
                eval_root=eval_root,
                accelerator=accelerator,
                config=config,
            )
            logger.info("Finish Intelligent-VBench evaluation!")
