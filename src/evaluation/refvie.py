"""
RefVIE-Bench evaluation using Gemini API for reference-guided video editing.

Modified from https://github.com/showlab/Kiwi-Edit/blob/main/eval_refvie_gemini.py
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

# ---------------------------------------------------------------------------
# Evaluation prompts per edit type
# ---------------------------------------------------------------------------

BACKGROUND_REFERENCE_PROMPT = """\
You are a data rater specializing in video background replacement grading. \
You will be given a **Reference Image**, an **Original Video** (foreground subject), \
and the **Edited Video** (result). Your task is to evaluate the background replacement \
effect on a 5-point scale from three perspectives, paying close attention to the \
preservation of the foreground subject and the fidelity to the reference image.

**Reference Fidelity & Preservation**
1. Background not changed, or the foreground subject is severely damaged/removed.
2. Background changed but bears no resemblance to the reference image; foreground edges are significantly cut off or distorted.
3. Background resembles the reference but lacks key details; foreground is mostly preserved but has noticeable missing parts or artifacts.
4. Background clearly matches the reference image structure and style; foreground subject is fully preserved with only minor edge errors.
5. Perfect execution: The background is an exact semantic and stylistic match to the reference image, and the foreground subject is preserved pixel-perfectly throughout the entire duration.

**Matting Quality & Temporal Stability**
1. Severe flickering; the background or foreground jitters erratically; distinct "boiling" artifacts on edges.
2. Obvious seams, halos, or "green screen" outlines around the subject; background moves unnaturally or freezes while the camera moves.
3. Edges are generally stable but soft/fuzzy; minor flickering in complex areas (e.g., hair, transparent objects); background stability is acceptable.
4. Clean edges with minimal temporal noise; background motion aligns well with camera movement; casual viewers notice no matting errors.
5. Completely seamless composition; hair/transparency details are perfectly matted; background and foreground interact with perfect temporal stability in every frame.

**Visual Harmony & Perspective**
1. Background looks like a flat 2D image pasted behind a 3D subject; severe perspective or lighting mismatch (e.g., shadows point wrong way).
2. Lighting clashes (e.g., sunny background, dark foreground); no depth integration; subject looks "floating."
3. Perspective and scale are roughly correct; lighting is neutral but doesn't explicitly match the new environment's ambience.
4. Good environmental integration; foreground lighting tones reflect the new background; cast shadows are present and mostly accurate.
5. Photorealistic integration: Depth of field, motion blur, lighting, and color grading of the foreground perfectly match the reference background; the composite looks like a single, raw video capture.

**The second and third score should no higher than first score!!!**

**Example Response Format:**
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Reference Fidelity & Preservation: A number from 1 to 5.
Matting Quality & Temporal Stability: A number from 1 to 5.
Visual Harmony & Perspective: A number from 1 to 5.

**editing instruction is : {prompt}**
**Below are the reference image, original video, and edited video:**
"""

SUBJECT_REFERENCE_PROMPT = """\
You are a data rater specializing in reference-guided object manipulation in videos. \
You will be given a **Reference Image** (the object to insert/swap), an **Original Video**, \
and the **Edited Video**. Your task is to evaluate the editing effect on a 5-point scale \
from three perspectives, specifically checking if the new object in the video matches \
the identity of the reference image.

**Identity Consistency & Compliance**
1. Object not swapped/added, or a completely unrelated object appears.
2. Object is changed, but looks nothing like the reference image (wrong color, shape, or class).
3. Object class is correct, but identity details (texture, specific markings, logos) differ significantly from the reference image.
4. High resemblance to the reference image; correct geometry and texture, with only minor variations in fine details.
5. Perfect identity transfer: The object in the video is indistinguishable from the reference image in terms of texture, structure, and style, while maintaining the correct pose for the scene.

**Temporal Consistency & Texture Fidelity**
1. The new object deforms, melts, or changes shape uncontrollably across frames.
2. Texture "swims" or flickers; resolution drops significantly compared to the rest of the video; object vanishes in some frames.
3. Object is stable in form, but texture details blur or shift slightly during motion; style looks somewhat pasted-on.
4. Object is structurally solid and texture is consistent; minor edge shimmer or noise visible only on close inspection.
5. Completely temporally coherent; the object maintains rigid structure (or appropriate flexibility) and consistent texture details in every single frame, exactly like a real object.

**Physical Integration & Tracking**
1. Object slides around (bad motion tracking); does not follow camera or scene movement; looks like a sticker on the screen.
2. Missing interactions: No shadows, reflections, or occlusion handling (e.g., object appears on top of things that should be in front of it).
3. Motion tracking is decent with slight drift; lighting is flat or generic; occlusion is roughly correct but imprecise.
4. Accurate tracking; lighting and shadows match the scene's direction and intensity; correct occlusion handling.
5. Physically flawless: Motion tracking, perspective changes, motion blur, shadows, reflections, and lighting interactions are indistinguishable from reality; the object feels physically present in the scene.

**The second and third score should no higher than first score!!!**

**Example Response Format:**
Brief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.
Identity Consistency & Compliance: A number from 1 to 5.
Temporal Consistency & Texture Fidelity: A number from 1 to 5.
Physical Integration & Tracking: A number from 1 to 5.

**editing instruction is : {prompt}**
**Below are the reference image, original video, and edited video:**
"""

EDIT_TYPE_TO_PROMPT = {
    "subject": SUBJECT_REFERENCE_PROMPT,
    "background": BACKGROUND_REFERENCE_PROMPT,
}

TASK_SCORE_DIMENSIONS = {
    "subject": [
        "Identity Consistency & Compliance",
        "Temporal Consistency & Texture Fidelity",
        "Physical Integration & Tracking",
    ],
    "background": [
        "Reference Fidelity & Preservation",
        "Matting Quality & Temporal Stability",
        "Visual Harmony & Perspective",
    ],
}


def generate_refvie_csv(dataset, eval_root: str, accelerator):
    """Collect generated video paths and metadata into per-dataset RefVIE CSVs."""
    fieldnames = [
        "edit_type", "prompt", "original_video", "ref_image", "edited_result_path",
    ]

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

            ref_paths = data_info.get("reference_image_paths", [])
            rows_by_dataset[dataset_name].append({
                "edit_type": data_info.get("edit_type", ""),
                "prompt": data_info.get(
                    "instruction", data_info.get("prompt", "")
                ),
                "original_video": data_info.get(
                    "source_video_path", data_info.get("original_video", "")
                ),
                "ref_image": ref_paths[0] if ref_paths else "",
                "edited_result_path": output_video_path,
            })

    # Write per-rank temporary CSVs
    for dataset_name, rows in rows_by_dataset.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        temp_csv_path = os.path.join(
            metrics_dir, f"refvie_input_rank{accelerator.process_index}.csv"
        )
        with open(temp_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    accelerator.wait_for_everyone()

    # Main process merges all per-rank CSVs
    if accelerator.is_main_process:
        for dataset_name in dataset.dataset_names:
            metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
            merged_rows = []
            for proc_idx in range(accelerator.num_processes):
                temp_path = os.path.join(
                    metrics_dir, f"refvie_input_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if merged_rows:
                csv_path = os.path.join(metrics_dir, "refvie_input.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(merged_rows)
                logging.info("Wrote %d rows to %s", len(merged_rows), csv_path)
            else:
                logging.warning(
                    "No valid RefVIE outputs for dataset %s", dataset_name
                )

    accelerator.wait_for_everyone()


def _extract_refvie_scores(response_text: str, edit_type: str):
    """
    Extract three dimension scores from Gemini response.

    Returns:
        Tuple of (scores_list, average) or (None, None) on parse failure.
    """
    dimensions = TASK_SCORE_DIMENSIONS.get(edit_type)
    if not dimensions:
        return None, None

    scores = []
    for dim in dimensions:
        match = re.search(re.escape(dim) + r":\s*(\d+\.?\d*)", response_text)
        if not match:
            return None, None
        value = float(match.group(1))
        if not 1 <= value <= 5:
            return None, None
        scores.append(value)

    return scores, round(sum(scores) / len(scores), 2)


def _call_gemini(
    ref_image_path: str,
    original_video_path: str,
    edited_video_path: str,
    prompt: str,
    gemini_url: str,
    gemini_headers: dict,
    max_retries: int = 5,
):
    """Call Gemini generateContent API with a reference image and two videos."""
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }

    for attempt in range(max_retries):
        try:
            parts = [{"text": prompt.strip()}]

            # Reference image
            if os.path.exists(ref_image_path):
                with open(ref_image_path, "rb") as img_f:
                    encoded_img = base64.b64encode(img_f.read()).decode("utf-8")
                mime = mime_map.get(
                    os.path.splitext(ref_image_path)[1].lower(), "image/jpeg"
                )
                parts.append({
                    "inline_data": {"mime_type": mime, "data": encoded_img}
                })
            else:
                parts.append({
                    "file_data": {
                        "mime_type": "image/jpeg",
                        "file_uri": ref_image_path,
                    }
                })

            # Original video + edited video
            for video_path in (original_video_path, edited_video_path):
                if os.path.exists(video_path):
                    with open(video_path, "rb") as vf:
                        encoded = base64.b64encode(vf.read()).decode("utf-8")
                    parts.append({
                        "inline_data": {
                            "mime_type": "video/mp4",
                            "data": encoded,
                        }
                    })
                else:
                    parts.append({
                        "file_data": {
                            "mime_type": "video/mp4",
                            "file_uri": video_path,
                        }
                    })

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
                "Gemini retry %d/%d: %s",
                attempt + 1,
                max_retries,
                str(result)[:200],
            )
            time.sleep(60)

        except Exception as exc:
            logging.warning(
                "Gemini retry %d/%d: %s", attempt + 1, max_retries, exc
            )
            time.sleep(60)

    return "ERROR: All retries exhausted"


def _score_single_refvie_sample(
    edit_type: str,
    prompt: str,
    original_video: str,
    ref_image: str,
    edited_result_path: str,
    original_video_root: str,
    gemini_url: str,
    gemini_headers: dict,
):
    """Score a single RefVIE sample via Gemini."""
    base_row = {
        "edit_type": edit_type,
        "prompt": prompt,
        "original_video": original_video,
        "ref_image": ref_image,
        "edited_result_path": edited_result_path,
    }
    error_row = lambda msg: {
        **base_row,
        "results": msg,
        "scores": "",
        "average": "ERROR",
    }

    # Resolve paths relative to data root
    resolved_original = original_video
    if original_video_root and not os.path.isabs(original_video):
        resolved_original = os.path.join(original_video_root, original_video)

    resolved_ref = ref_image
    if original_video_root and ref_image and not os.path.isabs(ref_image):
        resolved_ref = os.path.join(original_video_root, ref_image)

    if not os.path.exists(edited_result_path):
        return error_row(f"ERROR: Video not found: {edited_result_path}")
    if not os.path.exists(resolved_original):
        return error_row(f"ERROR: Video not found: {resolved_original}")
    if not os.path.exists(resolved_ref):
        return error_row(f"ERROR: Ref image not found: {resolved_ref}")

    system_prompt = EDIT_TYPE_TO_PROMPT.get(edit_type)
    if system_prompt is None:
        return error_row(f"ERROR: Unknown edit type: {edit_type}")

    try:
        response = _call_gemini(
            resolved_ref,
            resolved_original,
            edited_result_path,
            system_prompt.format(prompt=prompt),
            gemini_url,
            gemini_headers,
        )
        formatted_response = response.replace("\n", "\\n")
        scores, average = _extract_refvie_scores(response, edit_type)
        return {
            **base_row,
            "results": formatted_response,
            "scores": json.dumps(scores) if scores else "",
            "average": average if average is not None else "ERROR",
        }
    except Exception as exc:
        return error_row(f"ERROR: {exc}")


def _aggregate_refvie_scores(all_rows: list):
    """Compute per-edit-type dimension averages and overall averages."""
    type_dim_sums = defaultdict(lambda: defaultdict(float))
    type_counts = defaultdict(int)
    all_averages = []

    for row in all_rows:
        edit_type = row.get("edit_type", "")
        scores_str = row.get("scores", "")
        avg = row.get("average", "ERROR")

        if not scores_str or avg == "ERROR":
            continue
        try:
            scores = json.loads(scores_str)
        except (json.JSONDecodeError, TypeError):
            continue

        dimensions = TASK_SCORE_DIMENSIONS.get(edit_type, [])
        if len(scores) != len(dimensions):
            continue

        type_counts[edit_type] += 1
        for dim_name, score_val in zip(dimensions, scores):
            type_dim_sums[edit_type][dim_name] += score_val
        try:
            all_averages.append(float(avg))
        except (ValueError, TypeError):
            pass

    type_summaries = {}
    for edit_type, dims in TASK_SCORE_DIMENSIONS.items():
        count = type_counts.get(edit_type, 0)
        if count == 0:
            type_summaries[edit_type] = {
                "count": 0,
                "average": None,
                "dimension_averages": {},
            }
            continue
        dim_avgs = {
            d: round(type_dim_sums[edit_type][d] / count, 4) for d in dims
        }
        type_summaries[edit_type] = {
            "count": count,
            "average": round(sum(dim_avgs.values()) / len(dims), 4),
            "dimension_averages": dim_avgs,
        }

    return {
        "overall_average": (
            round(sum(all_averages) / len(all_averages), 4)
            if all_averages
            else None
        ),
        "type_summaries": type_summaries,
        "total_processed": len(all_rows),
        "total_valid": len(all_averages),
    }


def evaluate_on_refvie(dataset, eval_root: str, accelerator, config):
    """
    Full RefVIE-Bench evaluation pipeline.

    Requires ``config.model.evaluation.gemini`` with fields:
        - api_key: Bearer token for Authorization header
        - base_url: Gemini API base URL
        - model: Model identifier (default: gemini-2.5-pro-06-17)
    """
    # Step 1: Generate evaluation CSV
    generate_refvie_csv(
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
        "edit_type", "prompt", "original_video", "ref_image",
        "edited_result_path", "results", "scores", "average",
    ]

    # Step 2: Distributed Gemini scoring
    for dataset_name in dataset.dataset_names:
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        input_csv = os.path.join(metrics_dir, "refvie_input.csv")

        if not os.path.exists(input_csv):
            logging.warning(
                "No refvie_input.csv for dataset %s, skipping.", dataset_name
            )
            continue

        original_video_root = dataset_name_to_data_root.get(dataset_name, "")

        with open(input_csv, "r", encoding="utf-8-sig") as f:
            all_rows = list(csv.DictReader(f))

        temp_rows = []
        with accelerator.split_between_processes(
            list(range(len(all_rows)))
        ) as local_indices:
            for row_idx in local_indices:
                row = all_rows[row_idx]
                logging.info(
                    "[Rank %d] RefVIE scoring row %d/%d (type=%s)",
                    accelerator.process_index,
                    row_idx + 1,
                    len(all_rows),
                    row.get("edit_type", ""),
                )
                scored_row = _score_single_refvie_sample(
                    edit_type=row.get("edit_type", ""),
                    prompt=row.get("prompt", ""),
                    original_video=row.get("original_video", ""),
                    ref_image=row.get("ref_image", ""),
                    edited_result_path=row.get("edited_result_path", ""),
                    original_video_root=original_video_root,
                    gemini_url=gemini_url,
                    gemini_headers=gemini_headers,
                )
                temp_rows.append(scored_row)

        temp_csv_path = os.path.join(
            metrics_dir, f"refvie_scored_rank{accelerator.process_index}.csv"
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
                    metrics_dir, f"refvie_scored_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if not merged_rows:
                continue

            output_csv = os.path.join(metrics_dir, "refvie_scored.csv")
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            stats = _aggregate_refvie_scores(merged_rows)

            summary_path = os.path.join(metrics_dir, "refvie.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logging.info(
                "RefVIE-Bench [%s]: overall=%.4f, type_summaries=%s",
                dataset_name,
                stats.get("overall_average", 0) or 0,
                stats.get("type_summaries", {}),
            )

    accelerator.wait_for_everyone()
