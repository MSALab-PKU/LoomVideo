"""
Intelligent-VBench evaluation using Gemini API.

Modified from IntelligentVBench(https://github.com/Tencent-Hunyuan/OmniWeaving/tree/main/intelligentVBench)
"""

import os
import re
import csv
import json
import time
import base64
import logging
from collections import defaultdict

import requests


IMPLICIT_I2V_PROMPT = (
    "You are an expert data rater specializing in evaluating Image-to-Video (I2V) "
    "generation. You will be provided with an input image (serving as the first frame), "
    "a natural language instruction, and the generated video.\n"
    "Your task is to evaluate the generated video on a 5-point scale across three "
    "dimensions:\n\n"
    "1. The first score: Frame Consistency\n"
    "Objective: Evaluate whether the first frame of the generated video perfectly "
    "reconstructs the input image.\n"
    "- 5: Perfect Consistency. The first frame is an identical reconstruction.\n"
    "- 4: High Fidelity. Highly consistent with only negligible differences.\n"
    "- 3: Moderate Consistency. Recognizable but with visible shifts.\n"
    "- 2: Low Fidelity. Significant deviations from the input.\n"
    "- 1: Total Dissociation. Completely fails to use the input image.\n\n"
    "2. The second score: Instruction Following\n"
    "- 5: Perfect Alignment.\n"
    "- 4: Good Alignment.\n"
    "- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n"
    "- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n"
    "- 4: Good.\n"
    "- 3: Fair.\n"
    "- 2: Poor.\n"
    "- 1: Unacceptable.\n\n"
    "Example Response Format:\n"
    "You are required to return a dictionary structured as follows:\n"
    "{\n"
    '    "Frame Consistency": A number from 1 to 5.\n'
    '    "Instruction Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n'
    "}\n\n"
    "The instruction is: <input_prompt>\n"
    "This is the input image serving as the first frame:\n"
)

INTERPOLATIVE_DI2V_PROMPT = (
    "You are an expert data rater specializing in evaluating Dual-Image-to-Video "
    "(DI2V) interpolation generation. You will be provided with two input images "
    "(first frame and last frame), a natural language instruction, and the generated "
    "video.\nYour task is to evaluate the generated video on a 5-point scale across "
    "three dimensions:\n\n"
    "1. The first score: Frame Consistency\n"
    "Objective: Evaluate whether the first and last frames of the generated video "
    "perfectly anchor to the two input images.\n"
    "- 5: Perfect Consistency. Both frames are identical reconstructions.\n"
    "- 4: High Fidelity. Both frames highly consistent with only negligible differences.\n"
    "- 3: Moderate Consistency. Both frames recognizable but with visible shifts.\n"
    "- 2: Low Fidelity. One or both frames deviate significantly.\n"
    "- 1: Total Dissociation. Completely fails to use the input images.\n\n"
    "2. The second score: Instruction Following\n"
    "- 5: Perfect Alignment.\n"
    "- 4: Good Alignment.\n"
    "- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n"
    "- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n"
    "- 4: Good.\n"
    "- 3: Fair.\n"
    "- 2: Poor.\n"
    "- 1: Unacceptable.\n\n"
    "Example Response Format:\n"
    "You are required to return a dictionary structured as follows:\n"
    "{\n"
    '    "Frame Consistency": A number from 1 to 5.\n'
    '    "Instruction Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n'
    "}\n\n"
    "The instruction is: <input_prompt>\n"
    "This is the input image serving as the first frame:\n"
)

COMPOSITIONAL_MI2V_1SUBJECT_PROMPT = (
    "You are an expert data rater specializing in evaluating subject-driven video "
    "generation. You will be given a reference image containing a specific subject, "
    "a text prompt, and the generated video.\nYour task is to evaluate the generated "
    "video on a 5-point scale from three perspectives:\n\n"
    "1. The first score: Subject Consistency\n"
    "- 5: Perfect Preservation.\n"
    "- 4: High Preservation with only minor detail loss.\n"
    "- 3: Moderate Preservation with noticeable flaws.\n"
    "- 2: Low Preservation with severe drift.\n"
    "- 1: Complete Failure.\n\n"
    "2. The second score: Prompt Following\n"
    "- 5: Perfect Alignment.\n- 4: Good Alignment.\n- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Subject Consistency": A number from 1 to 5.\n'
    '    "Prompt Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The text prompt is: <input_prompt>\n"
    "This is the reference image containing a specific subject:\n"
)

COMPOSITIONAL_MI2V_1SUBJECT_WITH_BG_PROMPT = (
    "You are an expert data rater specializing in evaluating subject-and-background-"
    "conditioned video generation. You will be given a reference image containing a "
    "specific subject, a reference background image, a text prompt, and the generated "
    "video.\nYour task is to evaluate the generated video on a 5-point scale from "
    "three perspectives:\n\n"
    "1. The first score: Subject and Background Consistency\n"
    "- 5: Both subject and background are flawlessly maintained.\n"
    "- 4: High Preservation with only minor detail loss.\n"
    "- 3: Moderate Preservation with noticeable flaws.\n"
    "- 2: Low Preservation with severe drift.\n- 1: Complete Failure.\n\n"
    "2. The second score: Prompt Following\n"
    "- 5: Perfect Alignment.\n- 4: Good Alignment.\n- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Subject and Background Consistency": A number from 1 to 5.\n'
    '    "Prompt Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The text prompt is: <input_prompt>\n"
    "This is the first reference image containing a specific subject:\n"
)

COMPOSITIONAL_MI2V_MULTI_SUBJECTS_PROMPT = (
    "You are an expert data rater specializing in evaluating multi-subject-driven "
    "video generation. You will be given multiple reference images (each containing "
    "a specific subject), a text prompt, and the generated video.\nYour task is to "
    "evaluate the generated video on a 5-point scale from three perspectives:\n\n"
    "1. The first score: Multi-Subject Consistency\n"
    "- 5: All subjects are flawlessly maintained.\n"
    "- 4: High Preservation with only minor detail loss.\n"
    "- 3: Moderate Preservation with noticeable flaws.\n"
    "- 2: Low Preservation or identity confusion.\n- 1: Complete Failure.\n\n"
    "2. The second score: Prompt Following\n"
    "- 5: Perfect Alignment.\n- 4: Good Alignment.\n- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Multi-Subject Consistency": A number from 1 to 5.\n'
    '    "Prompt Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The text prompt is: <input_prompt>\n"
    "This is the first reference image containing a specific subject:\n"
)

COMPOSITIONAL_MI2V_MULTI_SUBJECTS_WITH_BG_PROMPT = (
    "You are an expert data rater specializing in evaluating multi-subject-and-"
    "background-conditioned video generation. You will be given multiple reference "
    "images containing specific subjects, a reference background image, a text "
    "prompt, and the generated video.\nYour task is to evaluate the generated video "
    "on a 5-point scale from three perspectives:\n\n"
    "1. The first score: Multi-Subject and Background Consistency\n"
    "- 5: All subjects and background are flawlessly maintained.\n"
    "- 4: High Preservation with only minor detail loss.\n"
    "- 3: Moderate Preservation with noticeable flaws.\n"
    "- 2: Low Preservation or severe drift.\n- 1: Complete Failure.\n\n"
    "2. The second score: Prompt Following\n"
    "- 5: Perfect Alignment.\n- 4: Good Alignment.\n- 3: Partial Alignment.\n"
    "- 2: Weak Alignment.\n- 1: No Alignment.\n\n"
    "3. The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Multi-Subject and Background Consistency": A number from 1 to 5.\n'
    '    "Prompt Following": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The text prompt is: <input_prompt>\n"
    "This is the first reference image containing a specific subject:\n"
)

TIV2V_LOCAL_CHANGE_PROMPT = (
    "You are an expert data rater specializing in grading video object replacement "
    "edits. You will be provided with an original video, a reference image, the "
    "edited video, and the corresponding editing instructions.\nYour task is to "
    "evaluate the editing performance on a 5-point scale across three key "
    "dimensions.\n\n"
    "The first score: Instruction Following\n"
    "- 5: Flawless execution.\n- 4: Correct replacement with minor deviations.\n"
    "- 3: Basic instruction followed but with noticeable errors.\n"
    "- 2: Attempted but poorly executed.\n- 1: Target was not replaced.\n\n"
    "The second score: Detail Preserving\n"
    "- 5: Perfect preservation of both new object identity and unedited regions.\n"
    "- 4: High preservation with minor issues.\n- 3: Moderate preservation.\n"
    "- 2: Low preservation.\n- 1: Complete failure.\n\n"
    "The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Instruction Following": A number from 1 to 5.\n'
    '    "Detail Preserving": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The editing instruction is: <edit_prompt>\n"
    "Below are the original video, reference image, and edited video:\n"
)

TIV2V_BACKGROUND_CHANGE_PROMPT = (
    "You are an expert data rater specializing in grading video background "
    "replacement edits. You will be provided with an original video, a reference "
    "image, the edited video, and the corresponding editing instructions.\nYour "
    "task is to evaluate the background change on a 5-point scale across three "
    "key dimensions.\n\n"
    "The first score: Instruction Following\n"
    "- 5: Flawless execution.\n- 4: Correct background with minor deviations.\n"
    "- 3: Basic instruction followed but with noticeable errors.\n"
    "- 2: Attempted but poorly executed.\n"
    "- 1: No change or unrelated background.\n\n"
    "The second score: Detail Preserving\n"
    "- 5: Perfect preservation of foreground and reference background identity.\n"
    "- 4: High preservation.\n- 3: Moderate preservation.\n"
    "- 2: Low preservation.\n- 1: Complete failure.\n\n"
    "The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Instruction Following": A number from 1 to 5.\n'
    '    "Detail Preserving": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The editing instruction is: <edit_prompt>\n"
    "Below are the original video, reference image, and edited video:\n"
)

TIV2V_LOCAL_ADD_PROMPT = (
    "You are an expert data rater specializing in grading video object addition "
    "edits. You will be provided with an original video, a reference image, the "
    "edited video, and the corresponding editing instructions.\nYour task is to "
    "evaluate the edit quality on a 5-point scale across three key dimensions.\n\n"
    "The first score: Instruction Following\n"
    "- 5: Flawless execution.\n- 4: Correct object added with minor deviations.\n"
    "- 3: Basic instruction followed but with noticeable errors.\n"
    "- 2: Attempted but poorly executed.\n- 1: No edit performed.\n\n"
    "The second score: Detail Preserving\n"
    "- 5: Perfect preservation of added object identity and unedited regions.\n"
    "- 4: High preservation.\n- 3: Moderate preservation.\n"
    "- 2: Low preservation.\n- 1: Complete failure.\n\n"
    "The third score: Overall Visual Quality\n"
    "- 5: Excellent.\n- 4: Good.\n- 3: Fair.\n- 2: Poor.\n- 1: Unacceptable.\n\n"
    "Example Response Format:\n{\n"
    '    "Instruction Following": A number from 1 to 5.\n'
    '    "Detail Preserving": A number from 1 to 5.\n'
    '    "Overall Visual Quality": A number from 1 to 5.\n}\n\n'
    "The editing instruction is: <edit_prompt>\n"
    "Below are the original video, reference image, and edited video:\n"
)

# Mapping from task subtype to (prompt_template, required_score_keys)
TASK_TYPE_CONFIG = {
    # SI2V subtypes
    "implicit_i2v": {
        "prompt": IMPLICIT_I2V_PROMPT,
        "score_keys": [
            "Frame Consistency",
            "Instruction Following",
            "Overall Visual Quality",
        ],
    },
    "interpolative_di2v": {
        "prompt": INTERPOLATIVE_DI2V_PROMPT,
        "score_keys": [
            "Frame Consistency",
            "Instruction Following",
            "Overall Visual Quality",
        ],
    },
    "compositional_mi2v_1subject": {
        "prompt": COMPOSITIONAL_MI2V_1SUBJECT_PROMPT,
        "prompt_with_bg": COMPOSITIONAL_MI2V_1SUBJECT_WITH_BG_PROMPT,
        "score_keys": [
            "Subject Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
        "score_keys_with_bg": [
            "Subject and Background Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
    },
    "compositional_mi2v_2subject": {
        "prompt": COMPOSITIONAL_MI2V_MULTI_SUBJECTS_PROMPT,
        "prompt_with_bg": COMPOSITIONAL_MI2V_MULTI_SUBJECTS_WITH_BG_PROMPT,
        "score_keys": [
            "Multi-Subject Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
        "score_keys_with_bg": [
            "Multi-Subject and Background Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
    },
    "compositional_mi2v_3subject": {
        "prompt": COMPOSITIONAL_MI2V_MULTI_SUBJECTS_PROMPT,
        "prompt_with_bg": COMPOSITIONAL_MI2V_MULTI_SUBJECTS_WITH_BG_PROMPT,
        "score_keys": [
            "Multi-Subject Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
        "score_keys_with_bg": [
            "Multi-Subject and Background Consistency",
            "Prompt Following",
            "Overall Visual Quality",
        ],
    },
    # TIV2V subtypes
    "local_change": {
        "prompt": TIV2V_LOCAL_CHANGE_PROMPT,
        "score_keys": [
            "Instruction Following",
            "Detail Preserving",
            "Overall Visual Quality",
        ],
    },
    "back_change": {
        "prompt": TIV2V_BACKGROUND_CHANGE_PROMPT,
        "score_keys": [
            "Instruction Following",
            "Detail Preserving",
            "Overall Visual Quality",
        ],
    },
    "local_add": {
        "prompt": TIV2V_LOCAL_ADD_PROMPT,
        "score_keys": [
            "Instruction Following",
            "Detail Preserving",
            "Overall Visual Quality",
        ],
    },
}


def _infer_si2v_subtype(data_info: dict) -> str:
    """
    Infer SI2V subtask type from save_path and reference image count.

    - save_path starting with "Implicit_I2V/" → implicit_i2v
    - save_path starting with "Interpolative_DI2V/" → interpolative_di2v
    - save_path starting with "Compositional_MI2V_*" → compositional_mi2v
    """
    save_path = data_info.get("save_path", "")
    ref_paths = data_info.get("reference_image_paths", [])

    if save_path.startswith("Implicit_I2V/"):
        return "implicit_i2v"
    if save_path.startswith("Interpolative_DI2V/"):
        return "interpolative_di2v"
    if save_path.startswith("Compositional_MI2V"):
        subject_count = sum(
            1 for p in ref_paths if "background" not in p.lower()
        )
        if subject_count <= 1:
            return "compositional_mi2v_1subject"
        if subject_count == 2:
            return "compositional_mi2v_2subject"
        return "compositional_mi2v_3subject"

    return "unknown"


def _encode_file_b64(file_path: str) -> str:
    """Read a file and return its base64-encoded content."""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_intelligent_vbench_csv(dataset, eval_root: str, accelerator):
    """Collect generated video paths and metadata into per-dataset CSVs."""
    all_indices = list(range(len(dataset)))
    local_indices = all_indices[accelerator.process_index :: accelerator.num_processes]

    rows_by_dataset = defaultdict(list)
    for idx in local_indices:
        data = dataset[idx]
        if data is None:
            continue

        dataset_name = data["dataset_name"]
        for gen_relative_path in data["gen_paths"]:
            output_video_path = os.path.join(
                eval_root, dataset_name, "outputs", gen_relative_path
            )
            if not os.path.exists(output_video_path):
                candidate_mp4 = os.path.splitext(output_video_path)[0] + ".mp4"
                if os.path.exists(candidate_mp4):
                    output_video_path = candidate_mp4
                else:
                    continue

            json_path = os.path.splitext(output_video_path)[0] + ".json"
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as f:
                data_info = json.load(f)

            task_type = _infer_si2v_subtype(data_info)
            if task_type == "unknown":
                task_type = data_info.get("edited_type", "unknown")

            rows_by_dataset[dataset_name].append({
                "task_type": task_type,
                "index": data_info.get("index", ""),
                "instruction": data_info.get("instruction", ""),
                "ref_image_paths": json.dumps(
                    data_info.get("reference_image_paths", [])
                ),
                "source_video": data_info.get("source_video_path", ""),
                "gen_video_path": output_video_path,
            })

    fieldnames = [
        "task_type", "index", "instruction", "ref_image_paths",
        "source_video", "gen_video_path",
    ]

    for dataset_name, rows in rows_by_dataset.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        temp_csv_path = os.path.join(
            metrics_dir, f"ivbench_input_rank{accelerator.process_index}.csv"
        )
        with open(temp_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        for dataset_name in dataset.dataset_names:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            merged_rows = []
            for proc_idx in range(accelerator.num_processes):
                temp_path = os.path.join(
                    metrics_dir, f"ivbench_input_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if merged_rows:
                csv_path = os.path.join(metrics_dir, "ivbench_input.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(merged_rows)
                logging.info(
                    "Wrote %d rows to %s", len(merged_rows), csv_path
                )
            else:
                logging.warning(
                    "No valid outputs found for dataset %s", dataset_name
                )

    accelerator.wait_for_everyone()


def _extract_scores(response_text: str, required_keys: list):
    """
    Extract numeric scores from Gemini JSON response.

    Returns:
        Tuple of (scores_list, average) or (None, None) on parse failure.
    """
    if not response_text:
        return None, None

    parsed = None
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(parsed, dict):
        return None, None

    scores = []
    for key in required_keys:
        if key not in parsed:
            return None, None
        try:
            score = max(1.0, min(5.0, float(parsed[key])))
            scores.append(score)
        except (TypeError, ValueError):
            return None, None

    average = round(sum(scores) / len(scores), 2)
    return scores, average


def _build_gemini_parts(
    task_type: str,
    instruction: str,
    ref_image_paths: list,
    gen_video_path: str,
    source_video_path: str,
    data_root: str,
):
    """
    Build Gemini generateContent parts for a single sample.

    Args:
        data_root: Local filesystem root where source data (reference images,
            source videos) are stored.

    Returns:
        Tuple of (parts_list, error_message_or_None).
    """
    config = TASK_TYPE_CONFIG.get(task_type)
    if config is None:
        return None, f"Unknown task type: {task_type}"

    # For compositional_mi2v tasks, select prompt variant based on
    # whether a background image is present
    has_background = any("background" in p.lower() for p in ref_image_paths)
    if task_type.startswith("compositional_mi2v") and has_background:
        prompt_template = config.get("prompt_with_bg", config["prompt"])
    else:
        prompt_template = config["prompt"]

    # Fill in the prompt placeholder
    if "<input_prompt>" in prompt_template:
        filled_prompt = prompt_template.replace("<input_prompt>", instruction)
    elif "<edit_prompt>" in prompt_template:
        filled_prompt = prompt_template.replace("<edit_prompt>", instruction)
    else:
        filled_prompt = prompt_template

    parts = [{"text": filled_prompt}]

    def _resolve(rel_path):
        """Resolve a relative path against data_root."""
        if not rel_path:
            return ""
        if os.path.isabs(rel_path):
            return rel_path
        return os.path.join(data_root, rel_path) if data_root else rel_path

    def _mime_for(path):
        return "image/png" if path.lower().endswith(".png") else "image/jpeg"

    # TIV2V tasks: source video → ref image → edited video
    if task_type in ("local_change", "back_change", "local_add"):
        source_path = _resolve(source_video_path)
        if not source_path or not os.path.exists(source_path):
            return None, f"Source video not found: {source_path}"
        parts.append({
            "inline_data": {
                "mime_type": "video/mp4",
                "data": _encode_file_b64(source_path),
            }
        })

        if ref_image_paths:
            ref_path = _resolve(ref_image_paths[0])
            if not os.path.exists(ref_path):
                return None, f"Reference image not found: {ref_path}"
            parts.append({"text": "This is the reference image:"})
            parts.append({
                "inline_data": {
                    "mime_type": _mime_for(ref_path),
                    "data": _encode_file_b64(ref_path),
                }
            })

        if not os.path.exists(gen_video_path):
            return None, f"Generated video not found: {gen_video_path}"
        parts.append({"text": "This is the video after editing:"})
        parts.append({
            "inline_data": {
                "mime_type": "video/mp4",
                "data": _encode_file_b64(gen_video_path),
            }
        })

    # SI2V implicit_i2v: ref image + generated video
    elif task_type == "implicit_i2v":
        if not ref_image_paths:
            return None, "No reference image for implicit_i2v"
        ref_path = _resolve(ref_image_paths[0])
        if not os.path.exists(ref_path):
            return None, f"Reference image not found: {ref_path}"
        parts.append({
            "inline_data": {
                "mime_type": _mime_for(ref_path),
                "data": _encode_file_b64(ref_path),
            }
        })

        if not os.path.exists(gen_video_path):
            return None, f"Generated video not found: {gen_video_path}"
        parts.append({"text": "This is the generated video:"})
        parts.append({
            "inline_data": {
                "mime_type": "video/mp4",
                "data": _encode_file_b64(gen_video_path),
            }
        })

    # SI2V interpolative_di2v: first_frame + last_frame + generated video
    elif task_type == "interpolative_di2v":
        if len(ref_image_paths) < 2:
            return None, "interpolative_di2v requires 2 reference images"
        first_path = _resolve(ref_image_paths[0])
        last_path = _resolve(ref_image_paths[1])
        if not os.path.exists(first_path):
            return None, f"First frame not found: {first_path}"
        if not os.path.exists(last_path):
            return None, f"Last frame not found: {last_path}"

        parts.append({
            "inline_data": {
                "mime_type": _mime_for(first_path),
                "data": _encode_file_b64(first_path),
            }
        })
        parts.append({
            "text": "This is the second input image serving as the last frame:"
        })
        parts.append({
            "inline_data": {
                "mime_type": _mime_for(last_path),
                "data": _encode_file_b64(last_path),
            }
        })

        if not os.path.exists(gen_video_path):
            return None, f"Generated video not found: {gen_video_path}"
        parts.append({"text": "This is the generated video:"})
        parts.append({
            "inline_data": {
                "mime_type": "video/mp4",
                "data": _encode_file_b64(gen_video_path),
            }
        })

    # SI2V compositional_mi2v: subject images (+ optional bg) + generated video
    elif task_type.startswith("compositional_mi2v"):
        subject_paths = [
            p for p in ref_image_paths if "background" not in p.lower()
        ]
        bg_paths = [
            p for p in ref_image_paths if "background" in p.lower()
        ]

        for img_idx, rel_path in enumerate(subject_paths):
            abs_path = _resolve(rel_path)
            if not os.path.exists(abs_path):
                return None, f"Subject image not found: {abs_path}"
            if img_idx > 0:
                ordinal = {1: "second", 2: "third", 3: "fourth"}.get(
                    img_idx, f"{img_idx + 1}th"
                )
                parts.append({
                    "text": f"This is the {ordinal} reference image "
                    "containing a specific subject:"
                })
            parts.append({
                "inline_data": {
                    "mime_type": _mime_for(abs_path),
                    "data": _encode_file_b64(abs_path),
                }
            })

        if bg_paths:
            bg_abs = _resolve(bg_paths[0])
            if not os.path.exists(bg_abs):
                return None, f"Background image not found: {bg_abs}"
            ordinal_bg = {1: "second", 2: "third", 3: "fourth"}.get(
                len(subject_paths), f"{len(subject_paths) + 1}th"
            )
            parts.append({
                "text": f"This is the {ordinal_bg} reference image serving "
                "as the reference background image:"
            })
            parts.append({
                "inline_data": {
                    "mime_type": _mime_for(bg_abs),
                    "data": _encode_file_b64(bg_abs),
                }
            })

        if not os.path.exists(gen_video_path):
            return None, f"Generated video not found: {gen_video_path}"
        parts.append({"text": "This is the generated video:"})
        parts.append({
            "inline_data": {
                "mime_type": "video/mp4",
                "data": _encode_file_b64(gen_video_path),
            }
        })

    else:
        return None, f"Unsupported task type for parts building: {task_type}"

    return parts, None


def _call_gemini(parts, gemini_url: str, gemini_headers: dict, max_retries: int = 5):
    """Call the Gemini generateContent API with pre-built parts."""
    for attempt in range(max_retries):
        try:
            payload = {
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 8192,
                    "responseModalities": ["TEXT"],
                },
            }
            response = requests.post(
                gemini_url,
                headers=gemini_headers,
                data=json.dumps(payload),
                timeout=120,
            )
            result = response.json()

            if response.status_code == 200:
                candidates = result.get("candidates", [])
                if candidates:
                    for part in candidates[0].get("content", {}).get(
                        "parts", []
                    ):
                        if "text" in part:
                            return part["text"]

            logging.warning(
                "Gemini retry %d/%d: status=%s, response=%s",
                attempt + 1,
                max_retries,
                response.status_code,
                str(result)[:300],
            )
            time.sleep(60)
        except Exception as exc:
            logging.warning(
                "Gemini retry %d/%d: %s", attempt + 1, max_retries, exc
            )
            time.sleep(60)

    return "ERROR: All retries exhausted"


def _score_single_sample(row: dict, data_root: str, gemini_url: str, gemini_headers: dict):
    """Score a single Intelligent-VBench sample via Gemini."""
    task_type = row.get("task_type", "")
    instruction = row.get("instruction", "")
    gen_video_path = row.get("gen_video_path", "")
    source_video = row.get("source_video", "")

    try:
        ref_image_paths = json.loads(row.get("ref_image_paths", "[]"))
    except (json.JSONDecodeError, TypeError):
        ref_image_paths = []

    base_row = {
        "task_type": task_type,
        "index": row.get("index", ""),
        "instruction": instruction,
        "ref_image_paths": row.get("ref_image_paths", ""),
        "source_video": source_video,
        "gen_video_path": gen_video_path,
    }
    error_result = lambda msg: {
        **base_row,
        "results": msg,
        "scores": "",
        "average": "ERROR",
    }

    config = TASK_TYPE_CONFIG.get(task_type)
    if config is None:
        return error_result(f"ERROR: Unknown task type: {task_type}")

    # Select appropriate score keys based on background presence
    has_background = any("background" in p.lower() for p in ref_image_paths)
    if task_type.startswith("compositional_mi2v") and has_background:
        active_score_keys = config.get(
            "score_keys_with_bg", config["score_keys"]
        )
    else:
        active_score_keys = config["score_keys"]

    parts, error = _build_gemini_parts(
        task_type=task_type,
        instruction=instruction,
        ref_image_paths=ref_image_paths,
        gen_video_path=gen_video_path,
        source_video_path=source_video,
        data_root=data_root,
    )
    if error:
        return error_result(f"ERROR: {error}")

    try:
        response_text = _call_gemini(parts, gemini_url, gemini_headers)
        formatted_response = response_text.replace("\n", "\\n")
        scores, average = _extract_scores(response_text, active_score_keys)

        result = {
            **base_row,
            "results": formatted_response,
            "scores": json.dumps(scores) if scores else "",
            "average": average if average is not None else "ERROR",
        }
        print(
            f"[IVBench] index={row.get('index', '')} | type={task_type} "
            f"| scores={scores} | avg={average} | video={gen_video_path}",
            flush=True,
        )
        return result
    except Exception as exc:
        logging.warning(
            "[IVBench] index=%s | type=%s | ERROR: %s",
            row.get("index", ""),
            task_type,
            exc,
        )
        return error_result(f"ERROR: {exc}")


def _aggregate_scores(all_rows: list):
    """Compute per-task-type dimension averages and overall average."""
    type_dim_sums = defaultdict(lambda: defaultdict(float))
    type_counts = defaultdict(int)
    all_averages = []

    for row in all_rows:
        task_type = row.get("task_type", "")
        scores_str = row.get("scores", "")
        avg = row.get("average", "ERROR")

        if not scores_str or avg == "ERROR":
            continue
        try:
            scores = json.loads(scores_str)
        except (json.JSONDecodeError, TypeError):
            continue

        config = TASK_TYPE_CONFIG.get(task_type)
        if config is None:
            continue
        expected_num_dims = len(config["score_keys"])
        if len(scores) != expected_num_dims:
            continue

        canonical_dims = config["score_keys"]
        type_counts[task_type] += 1
        for dim_name, score_val in zip(canonical_dims, scores):
            type_dim_sums[task_type][dim_name] += score_val
        try:
            all_averages.append(float(avg))
        except (ValueError, TypeError):
            pass

    type_summaries = {}
    for task_type, task_config in TASK_TYPE_CONFIG.items():
        count = type_counts.get(task_type, 0)
        dims = task_config["score_keys"]
        if count == 0:
            type_summaries[task_type] = {
                "count": 0,
                "average": None,
                "dimension_averages": {},
            }
            continue
        dim_avgs = {
            d: round(type_dim_sums[task_type][d] / count, 4) for d in dims
        }
        type_summaries[task_type] = {
            "count": count,
            "average": round(sum(dim_avgs.values()) / len(dims), 4),
            "dimension_averages": dim_avgs,
        }

    # Weighted average for TIV2V subtypes
    tiv2v_types = ["local_change", "back_change", "local_add"]
    tiv2v_dims = [
        "Instruction Following",
        "Detail Preserving",
        "Overall Visual Quality",
    ]
    tiv2v_total_count = sum(type_counts.get(t, 0) for t in tiv2v_types)
    tiv2v_weighted_average = None
    if tiv2v_total_count > 0:
        tiv2v_dim_avgs = {}
        for dim in tiv2v_dims:
            weighted_sum = sum(type_dim_sums[t][dim] for t in tiv2v_types)
            tiv2v_dim_avgs[dim] = round(weighted_sum / tiv2v_total_count, 4)
        tiv2v_weighted_average = {
            "count": tiv2v_total_count,
            "average": round(
                sum(tiv2v_dim_avgs.values()) / len(tiv2v_dims), 4
            ),
            "dimension_averages": tiv2v_dim_avgs,
        }

    return {
        "overall_average": (
            round(sum(all_averages) / len(all_averages), 4)
            if all_averages
            else None
        ),
        "type_summaries": type_summaries,
        "tiv2v_weighted_average": tiv2v_weighted_average,
        "total_processed": len(all_rows),
        "total_valid": len(all_averages),
    }


def evaluate_on_intelligent_vbench(
    dataset, eval_root: str, accelerator, config
):
    """
    Full Intelligent-VBench evaluation pipeline (distributed).

    Requires ``config.model.evaluation.gemini`` with fields:
        - api_key: Bearer token for Authorization header
        - base_url: Gemini API base URL
        - model: Model identifier (default: gemini-2.5-pro-06-17)
    """
    # Step 1: Generate evaluation CSV
    generate_intelligent_vbench_csv(
        dataset=dataset, eval_root=eval_root, accelerator=accelerator
    )

    gemini_config = getattr(config.model.evaluation, "gemini", None)
    if gemini_config is None:
        logging.error(
            "Gemini config not found at config.model.evaluation.gemini. "
            "Required fields: base_url, api_key"
        )
        accelerator.wait_for_everyone()
        return

    gemini_base_url = getattr(gemini_config, "base_url", "")
    gemini_ak = getattr(gemini_config, "api_key", "")
    gemini_model = getattr(
        gemini_config, "model", "gemini-2.5-pro-06-17"
    )

    # Resolve source data root from dataset config (local filesystem)
    dataset_name_to_data_root = {
        name: getattr(info, "data_root", "")
        for name, info in zip(dataset.dataset_names, dataset.dataset_list)
    }

    gemini_url = (
        f"{gemini_base_url.rstrip('/')}/models/{gemini_model}:generateContent"
    )
    gemini_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gemini_ak}",
    }

    scored_fieldnames = [
        "task_type", "index", "instruction", "ref_image_paths",
        "source_video", "gen_video_path", "results", "scores", "average",
    ]

    # Step 2: Distributed Gemini scoring
    for dataset_name in dataset.dataset_names:
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        input_csv = os.path.join(metrics_dir, "ivbench_input.csv")

        if not os.path.exists(input_csv):
            logging.warning(
                "No ivbench_input.csv for dataset %s, skipping.", dataset_name
            )
            continue

        data_root = dataset_name_to_data_root.get(dataset_name, "")

        with open(input_csv, "r", encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))

        temp_rows = []
        with accelerator.split_between_processes(
            list(range(len(all_rows)))
        ) as local_indices:
            for row_idx in local_indices:
                row = all_rows[row_idx]
                logging.info(
                    "[Rank %d] IVBench scoring row %d/%d (type=%s)",
                    accelerator.process_index,
                    row_idx + 1,
                    len(all_rows),
                    row.get("task_type", ""),
                )
                scored_row = _score_single_sample(
                    row=row,
                    data_root=data_root,
                    gemini_url=gemini_url,
                    gemini_headers=gemini_headers,
                )
                temp_rows.append(scored_row)

        temp_csv_path = os.path.join(
            metrics_dir, f"ivbench_scored_rank{accelerator.process_index}.csv"
        )
        with open(temp_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
            writer.writeheader()
            writer.writerows(temp_rows)

    # Step 3: Synchronise and merge
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        for dataset_name in dataset.dataset_names:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            merged_rows = []
            for proc_idx in range(accelerator.num_processes):
                temp_path = os.path.join(
                    metrics_dir, f"ivbench_scored_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if not merged_rows:
                logging.warning("No scored rows for %s", dataset_name)
                continue

            scored_csv_path = os.path.join(metrics_dir, "ivbench_scored.csv")
            with open(scored_csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            summary = _aggregate_scores(merged_rows)
            summary_path = os.path.join(
                metrics_dir, "intelligent_vbench.json"
            )
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            logging.info(
                "Intelligent-VBench %s: overall_average=%s, total_valid=%d/%d",
                dataset_name,
                summary["overall_average"],
                summary["total_valid"],
                summary["total_processed"],
            )

    accelerator.wait_for_everyone()
