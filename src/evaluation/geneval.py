# Modified from GenEval (https://github.com/djghosh13/geneval)

import os
import json
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch

import mmdet
from mmdet.apis import inference_detector, init_detector

import open_clip
from clip_benchmark.metrics import zeroshot_classification as zsc

zsc.tqdm = lambda it, *args, **kwargs: it

GENEVAL_LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "libs", "geneval")
OBJECT_NAMES_PATH = os.path.join(GENEVAL_LIB_DIR, "evaluation", "object_names.txt")
METADATA_PATH = os.path.join(GENEVAL_LIB_DIR, "prompts", "evaluation_metadata.jsonl")

THRESHOLD = 0.3
COUNTING_THRESHOLD = 0.9
MAX_OBJECTS = 16
NMS_THRESHOLD = 1.0
POSITION_THRESHOLD = 0.1
COLORS = [
    "red", "orange", "yellow", "green", "blue",
    "purple", "pink", "brown", "black", "white",
]
BGCOLOR = "#999"


class GenevalEvaluator:
    """Loads Mask2Former + OpenCLIP once and exposes a single ``evaluate_image`` method."""

    def __init__(self, model_path, device, model_config=None, clip_path=None):
        # Object detector (Mask2Former)
        if model_config is None:
            model_config = os.path.join(
                os.path.dirname(mmdet.__file__),
                "../configs/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
            )
        self.object_detector = init_detector(model_config, model_path, device=device)
        self.object_detector.float()  # ms_deform_attn CUDA kernel does not support BFloat16

        # CLIP for color classification
        if clip_path is not None:
            self.clip_model, _, self.transform = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained=clip_path, device=device, weights_only=False,
            )
        else:
            self.clip_model, _, self.transform = open_clip.create_model_and_transforms(
                "ViT-L-14", pretrained="openai", device=device,
            )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")

        # COCO class names
        with open(OBJECT_NAMES_PATH) as fp:
            self.classnames = [line.strip() for line in fp]

        self.device = device
        self._color_classifiers = {}

    def evaluate_image(self, filepath, metadata):
        """Run detection + compositional evaluation on one image; return result dict."""
        detected = self._detect_objects(filepath, metadata)
        image = ImageOps.exif_transpose(Image.open(filepath))
        is_correct, reason = self._check_requirements(image, detected, metadata)

        return {
            "tag": metadata["tag"],
            "correct": is_correct,
            "reason": reason,
            "metadata": json.dumps(metadata),
            "details": json.dumps({
                key: [box.tolist() for box, _ in value]
                for key, value in detected.items()
            }),
        }

    def _detect_objects(self, filepath, metadata):
        """Run Mask2Former object detection."""
        with torch.autocast("cuda", enabled=False), torch.autocast("cpu", enabled=False):
            result = inference_detector(self.object_detector, filepath)
        bbox = result[0] if isinstance(result, tuple) else result
        segm = result[1] if isinstance(result, tuple) and len(result) > 1 else None

        confidence_threshold = (
            THRESHOLD if metadata["tag"] != "counting" else COUNTING_THRESHOLD
        )
        detected = {}

        for index, classname in enumerate(self.classnames):
            ordering = np.argsort(bbox[index][:, 4])[::-1]
            ordering = ordering[bbox[index][ordering, 4] > confidence_threshold]
            ordering = ordering[:MAX_OBJECTS].tolist()
            detected[classname] = []
            while ordering:
                max_obj = ordering.pop(0)
                detected[classname].append(
                    (bbox[index][max_obj], None if segm is None else segm[index][max_obj])
                )
                ordering = [
                    obj
                    for obj in ordering
                    if NMS_THRESHOLD == 1
                    or self._compute_iou(bbox[index][max_obj], bbox[index][obj])
                    < NMS_THRESHOLD
                ]
            if not detected[classname]:
                del detected[classname]

        return detected

    def _check_requirements(self, image, objects, metadata):
        """Check whether detected objects satisfy the GenEval metadata requirements."""
        correct = True
        reason = []
        matched_groups = []

        for req in metadata.get("include", []):
            classname = req["class"]
            matched = True
            found_objects = objects.get(classname, [])[:req["count"]]

            if len(found_objects) < req["count"]:
                correct = matched = False
                reason.append(
                    f"expected {classname}>={req['count']}, found {len(found_objects)}"
                )
            else:
                if "color" in req:
                    colors = self._classify_colors(image, found_objects, classname)
                    if colors.count(req["color"]) < req["count"]:
                        correct = matched = False
                        reason.append(
                            f"expected {req['color']} {classname}>={req['count']}, found "
                            + f"{colors.count(req['color'])} {req['color']}; and "
                            + ", ".join(
                                f"{colors.count(c)} {c}" for c in COLORS if c in colors
                            )
                        )
                if "position" in req and matched:
                    expected_rel, target_group = req["position"]
                    if matched_groups[target_group] is None:
                        correct = matched = False
                        reason.append(
                            f"no target for {classname} to be {expected_rel}"
                        )
                    else:
                        for obj in found_objects:
                            for target_obj in matched_groups[target_group]:
                                true_rels = self._relative_position(obj, target_obj)
                                if expected_rel not in true_rels:
                                    correct = matched = False
                                    reason.append(
                                        f"expected {classname} {expected_rel} target, found "
                                        + f"{' and '.join(true_rels)} target"
                                    )
                                    break
                            if not matched:
                                break

            matched_groups.append(found_objects if matched else None)

        for req in metadata.get("exclude", []):
            classname = req["class"]
            if len(objects.get(classname, [])) >= req["count"]:
                correct = False
                reason.append(
                    f"expected {classname}<{req['count']}, found {len(objects[classname])}"
                )

        return correct, "\n".join(reason)

    def _classify_colors(self, image, bboxes, classname):
        """Classify detected object colors using CLIP zero-shot classification."""
        if classname not in self._color_classifiers:
            self._color_classifiers[classname] = zsc.zero_shot_classifier(
                self.clip_model,
                self.tokenizer,
                COLORS,
                [
                    f"a photo of a {{c}} {classname}",
                    f"a photo of a {{c}}-colored {classname}",
                    f"a photo of a {{c}} object",
                ],
                self.device,
            )
        dataloader = torch.utils.data.DataLoader(
            _ImageCrops(image, bboxes, self.transform),
            batch_size=16,
            num_workers=4,
        )
        with torch.no_grad():
            pred, _ = zsc.run_classification(
                self.clip_model,
                self._color_classifiers[classname],
                dataloader,
                self.device,
            )
            return [COLORS[idx.item()] for idx in pred.argmax(1)]

    @staticmethod
    def _compute_iou(box_a, box_b):
        """Compute intersection-over-union of two bounding boxes."""
        area_fn = lambda box: max(box[2] - box[0] + 1, 0) * max(box[3] - box[1] + 1, 0)
        inter = area_fn([
            max(box_a[0], box_b[0]),
            max(box_a[1], box_b[1]),
            min(box_a[2], box_b[2]),
            min(box_a[3], box_b[3]),
        ])
        union = area_fn(box_a) + area_fn(box_b) - inter
        return inter / union if union else 0

    @staticmethod
    def _relative_position(obj_a, obj_b):
        """Determine spatial relationship between two detected objects."""
        boxes = np.array([obj_a[0], obj_b[0]])[:, :4].reshape(2, 2, 2)
        center_a, center_b = boxes.mean(axis=-2)
        dim_a, dim_b = np.abs(np.diff(boxes, axis=-2))[..., 0, :]
        offset = center_a - center_b
        revised = (
            np.maximum(np.abs(offset) - POSITION_THRESHOLD * (dim_a + dim_b), 0)
            * np.sign(offset)
        )
        if np.all(np.abs(revised) < 1e-3):
            return set()
        dx, dy = revised / np.linalg.norm(offset)
        relations = set()
        if dx < -0.5:
            relations.add("left of")
        if dx > 0.5:
            relations.add("right of")
        if dy < -0.5:
            relations.add("above")
        if dy > 0.5:
            relations.add("below")
        return relations


class _ImageCrops(torch.utils.data.Dataset):
    """Cropped object regions fed to CLIP for color classification."""

    def __init__(self, image, objects, transform):
        self._image = image.convert("RGB")
        self._blank = Image.new("RGB", image.size, color=BGCOLOR)
        self._objects = objects
        self._transform = transform

    def __len__(self):
        return len(self._objects)

    def __getitem__(self, index):
        box, mask = self._objects[index]
        if mask is not None:
            assert tuple(self._image.size[::-1]) == tuple(mask.shape)
            image = Image.composite(self._image, self._blank, Image.fromarray(mask))
        else:
            image = self._image
        return (self._transform(image.crop(box[:4])), 0)


def evaluate_on_geneval(dataset, eval_root, accelerator, config):
    """Evaluate generated images using the GenEval benchmark."""
    device = str(accelerator.device)

    # Resolve model paths from config
    geneval_config = getattr(config.model.evaluation, "geneval", None)
    model_path = geneval_config.model_path if geneval_config else "./"
    model_config_path = (
        getattr(geneval_config, "model_config_path", None) if geneval_config else None
    )
    clip_path = (
        getattr(geneval_config, "clip_path", None) if geneval_config else None
    )

    evaluator = GenevalEvaluator(model_path, device, model_config_path, clip_path)

    # Build prompt -> metadata lookup
    generation_prefix = "Generate an image: "
    with open(METADATA_PATH) as fp:
        prompt_to_metadata = {
            f"{generation_prefix}{m['prompt']}": m
            for m in (json.loads(line) for line in fp)
        }

    accelerator.wait_for_everyone()

    temp_rows = defaultdict(list)

    with accelerator.split_between_processes(list(range(len(dataset)))) as local_indices:
        for idx in local_indices:
            data = dataset[idx]
            if data is None:
                continue

            dataset_name = data["dataset_name"]
            instruction = data["instruction"]
            metadata = prompt_to_metadata.get(instruction)
            if metadata is None:
                warnings.warn(
                    f"GenEval metadata not found for prompt: '{instruction}', skipping."
                )
                continue

            output_dir = os.path.join(eval_root, dataset_name, "outputs")
            for image_path in data["gen_paths"]:
                if not image_path.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    continue
                full_image_path = (
                    os.path.join(output_dir, image_path)
                    if not os.path.isabs(image_path)
                    else image_path
                )
                if not os.path.isfile(full_image_path):
                    continue

                result = evaluator.evaluate_image(full_image_path, metadata)
                temp_rows[dataset_name].append({
                    "gen_path": full_image_path,
                    "instruction": instruction,
                    **result,
                })

    # Write per-rank temporary CSVs
    for dataset_name, rows in temp_rows.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        temp_path = os.path.join(
            metrics_dir, f"geneval_each_rank{accelerator.process_index}.csv"
        )
        pd.DataFrame(rows).to_csv(temp_path, index=False)

    accelerator.wait_for_everyone()

    # Main process: merge and compute summary scores
    if accelerator.is_main_process:
        for dataset_name in dataset.dataset_names:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            temp_dfs = []
            for proc_idx in range(accelerator.num_processes):
                temp_path = os.path.join(
                    metrics_dir, f"geneval_each_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    temp_dfs.append(pd.read_csv(temp_path))
                    os.remove(temp_path)

            if not temp_dfs:
                continue

            merged_df = pd.concat(temp_dfs, ignore_index=True)
            merged_df.to_csv(
                os.path.join(metrics_dir, "geneval_each.csv"), index=False
            )

            summary = {}
            task_scores = []
            for tag, task_df in merged_df.groupby("tag", sort=False):
                tag_accuracy = float(task_df["correct"].mean())
                task_scores.append(tag_accuracy)
                summary[f"geneval_{tag}_accuracy"] = tag_accuracy

            summary["geneval_overall_score"] = float(np.mean(task_scores))
            summary["geneval_total_images"] = int(len(merged_df))
            summary["geneval_correct_images_pct"] = float(merged_df["correct"].mean())

            with open(
                os.path.join(metrics_dir, "geneval.json"), "w", encoding="utf-8"
            ) as fp:
                json.dump(summary, fp, indent=2)

    accelerator.wait_for_everyone()
