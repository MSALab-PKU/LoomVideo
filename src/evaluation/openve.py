"""
OpenVE-Bench evaluation using Gemini API for video editing quality assessment.

Modified from OpenVE-Bench (https://github.com/OpenVE-Team/OpenVE-3M/blob/main/OpenVE-Bench/gemini_benchmark.py)
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


Global_Style = """\nYou are a data rater specializing in grading global style transfer edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 Target style absent; original video unchanged or corrupted.\n2 Style shows in isolated areas only or mixed with unrelated styles; main content distorted.\n3 Key traits (palette, brushwork/texture, mood) present but patchy, inconsistent, or only partially cover the video.\n4 Style covers the full video consistently; only small local mismatches remain (e.g., a few frames revert).\n5 Full, faithful transfer: colour, texture, brushwork, and lighting faithfully match the requested style in every frame throughout the video.\n\nTemporal & Visual Coherence\n1 Massive flickering, strobing, or style "blinking" between frames; visually unwatchable.\n2 Obvious temporal instability: style oscillates strongly between frames; clear "boiling" or "shimmering" in textures.\n3 Style is generally maintained but with noticeable per-frame fluctuation (e.g., colour saturation pulses, texture shifts).\n4 Very stable style application; only slight, hard-to-spot frame-to-frame variation upon close inspection.\n5 Perfectly temporally stable; the style feels baked into the footage with zero detectable flickering or jitter.\n\nContent Preservation & Physical Plausibility\n1 Original scene barely recognisable; major objects or layout lost/distorted; new artefacts dominate.\n2 Main subject recognisable, but size, perspective, or key parts clearly wrong/missing; background heavily altered.\n3 Overall structure and spatial relations correct; some local warping, minor omissions, or small physical errors.\n4 Nearly all geometry and detail intact; only slight, non-distracting deformation; plausible lighting/shadows.\n5 All objects and spatial relations perfectly maintained; stylistic changes are purely aesthetic with zero structural distortion.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nTemporal & Visual Coherence: A number from 1 to 5.\nContent Preservation & Physical Plausibility: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

Creative_Edit = """\nYou are a data rater specializing in grading creative video edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the creative edit quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 The edit does not reflect the instruction at all; the video is unchanged or corrupted.\n2 The edit only vaguely resembles the instruction; key elements are wrong or missing.\n3 The main idea of the instruction is captured, but important details are incorrect or incomplete.\n4 The instruction is largely fulfilled with only minor deviations from the creative intent.\n5 The video creatively interprets and executes the instruction throughout the video's duration, fully achieving the intended creative goal.\n\nTemporal & Visual Coherence\n1 Massive flickering, strobing, or artifacts that make the video unwatchable; edited elements are completely disjointed from the scene.\n2 Obvious temporal inconsistency (e.g., style flickers on/off), clear visual boundaries or seams; mismatched color/lighting between frames.\n3 The edit is mostly stable, but with noticeable "boiling" or "shimmering" in textures/styles; minor jitter or softness on edges.\n4 The edit is very stable and well-integrated; only slight, hard-to-spot artifacts or flickering are present, motion is smooth.\n5 Perfectly stable and seamless integration; the edit feels like part of the original footage with no detectable flickering, jitter, or discontinuities.\n\nPhysical Plausibility & Detail Preservation\n1 Complete break from physical laws; added objects have no correct lighting/shadows, move unnaturally; original video details are heavily degraded.\n2 Major physical inconsistencies; shadows/reflections are static or move incorrectly; motion of edits doesn't match camera movement; original background is warped.\n3 Physics and lighting are generally believable but with minor flaws (e.g., a shadow is slightly off); unedited parts of the video are mostly preserved.\n4 Edited elements interact realistically with the scene's lighting, motion, and perspective; original video details are well-preserved.\n5 High degree of physical realism and integration; motion, lighting, and physics of the edits are indistinguishable from a real recording; original details are perfectly maintained.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nTemporal & Visual Coherence: A number from 1 to 5.\nPhysical Plausibility & Detail Preservation: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

Camera_Edit = """\nYou are a data rater specializing in grading camera shot type alteration edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the camera shot change on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 The shot type is not changed, or changed to a completely wrong type (e.g., requested close-up, but got a long shot).\n2 The direction of the shot change is correct (e.g., zoomed in for a close-up), but the degree is wrong (e.g., a medium shot instead of a close-up).\n3 The shot type is generally correct, but the framing is poor, cutting off important parts of the subject or being poorly centered.\n4 The shot type and framing are correct, with only minor inaccuracies in composition.\n5 The video is perfectly transformed into the requested shot type (long, medium, or close-up) with ideal framing of the subject.\n\nVisual Quality & Stability\n1 Massive distortion, glitches, warping, or heavy noise; the edited video is unusable.\n2 Significant and distracting jitter, shimmering, or warping is visible throughout the video, making the shot feel unstable.\n3 Minor but noticeable visual flaws, such as slight edge distortion or a subtle "breathing" effect in the frame.\n4 The video is stable and clear, with only very slight, almost unnoticeable artifacts upon close inspection.\n5 The resulting shot is perfectly stable and clear, with no digital artifacts, distortion, or jitter. It looks as if it were originally filmed with that shot type.\n\nConsistency & Detail Fidelity\n1 The subject, background, or action in the edited video is completely different from the original video; a total failure of consistency.\n2 The main subject is the same, but their action, the background, or the lighting is drastically and illogically changed compared to the original video.\n3 The scene is generally consistent, but there are noticeable continuity errors (e.g., an object disappears, the subject's pose changes unnaturally).\n4 The subject, action, and environment are highly consistent with the original video. Original details are well-preserved with only minor, hard-to-spot discrepancies.\n5 Perfect consistency; the edited video perfectly preserves the subject, lighting, background, and continuity of action from the original video, creating the illusion of the same scene captured from a different camera position.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Quality & Stability: A number from 1 to 5.\nConsistency & Detail Fidelity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

Local_Change = """\nYou are a data rater specializing in grading video replacement edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives, paying close attention to temporal consistency (how the edit holds up over time and with motion).\n\nPrompt Compliance\n1 Target not replaced, or an unrelated object/part of the video edited.\n2 Only part of the target replaced (e.g., in only a few frames), or wrong class/description used.\n3 Target largely replaced but other objects altered, remnants visible across frames, or count/position clearly wrong.\n4 Correct object fully replaced for the entire duration; only minor attribute errors (colour, size, etc.).\n5 Perfect replacement: all and only the specified objects replaced for the entire duration; new objects' class, number, position, scale, pose, motion and detail exactly match the prompt.\n\nVisual Naturalness & Temporal Stability\n1 Video heavily broken or new object deformed / flickers uncontrollably / jitters erratically.\n2 Obvious seams/edges that flicker or move unnaturally; strong mismatch in resolution or colour that is inconsistent across frames; background not restored or is unstable.\n3 Basic style similar, but lighting or palette clashes are inconsistent as the video plays; fuzzy edges, noise or minor flickering/jittering are noticeable.\n4 Style almost uniform and stable; tiny temporal artefacts (e.g., edge shimmer) visible only on close, frame-by-frame inspection; casual viewers see no edit.\n5 Completely seamless and temporally stable; new objects blend fully with the scene in every frame, edit area undetectable.\n\nPhysical & Motion Integrity\n1 Floating or sliding unnaturally (poor motion tracking), severe perspective/light errors inconsistent with camera/object movement; background heavily warped or unstable.\n2 Missing or static shadows/reflections that do not move with the object/light; poor occlusion; new object's motion clearly mismatches scene motion.\n3 Lighting, perspective and interactions mostly correct but with minor inconsistencies over time; motion tracking has small, tolerable drifts.\n4 New object's motion is well-tracked and it interacts realistically with the scene (shadows, reflections) and preserves existing details throughout the video.\n5 Physically and dynamically flawless: motion, perspective, shadows, and reflections are perfectly integrated and move correctly with the scene and camera in every frame; background untouched and stable.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Naturalness & Temporal Stability: A number from 1 to 5.\nPhysical & Motion Integrity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

Background_Change = """\nYou are a data rater specializing in grading video background editing. You will be given two videos (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:\n\nInstruction Compliance\n1 No change, or background unrelated to prompt, or foreground also replaced/distorted.\n2 Background partly replaced or wrong style/content; foreground noticeably altered.\n3 Main background replaced but elements missing/extra, or faint spill onto subject edges.\n4 Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).\n5 Background exactly matches prompt (content, style, placement); all foreground pixels untouched.\n\nVisual & Temporal Seamlessness (Edge, Blend & Stability)\n1 Large tearing, posterisation, or significant temporal artifacts like flickering, jittering edges; edit area obvious at a glance.\n2 Clear cut-out halos, colour-resolution gap, or obvious edge 'boiling' (instability) over time.\n3 Blend acceptable but visible on closer look: slight edge blur, or minor temporal instability (e.g., shimmer).\n4 Nearly invisible seams; edges are stable across motion, textures aligned, only minor issues when zoomed in.\n5 Indistinguishable composite: edges, textures, resolution and colour grading are perfectly continuous and stable throughout the video's duration.\n\nPhysical Consistency (Lighting, Perspective, Motion & Depth)\n1 Severe mismatch: wrong horizon, conflicting light, floating subject, or background remains static during camera movement (no parallax).\n2 Noticeable inconsistencies in light or scale; incorrect perspective shifts during motion.\n3 Overall believable; small errors in shadow, perspective, or minor motion tracking flaws.\n4 Lighting, scale, and depth well matched; background perspective and scale track convincingly with camera motion.\n5 Physically flawless: foreground and new background share coherent light, shadows, perspective, and atmospheric depth throughout all subject and camera motion, enhancing overall realism.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nInstruction Compliance: A number from 1 to 5.\nVisual & Temporal Seamlessness: A number from 1 to 5.\nPhysical Consistency: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

Local_Remove = """\nYou are a data rater specializing in grading video object removal editing. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the edit quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 No edit performed, the video is corrupted, or the edit is completely wrong.\n2 Wrong object/class removed, or target only partially removed, or an unrelated object is also removed.\n3 Correct object removed, but with significant errors: unintended objects are also removed, OR significant fragments/ghosting of the target remain.\n4 The correct object is removed; only minor issues like a few tiny fragments remaining or tiny, unintended background items being affected.\n5 Perfect: All and only the requested objects are removed as instructed; every other element is untouched.\n\nVisual & Temporal Naturalness\n1 Video is badly broken, full of artefacts, or shows severe flickering/jittering throughout.\n2 Obvious erase marks or "smudges"; the inpainted background's style, resolution, or palette strongly mismatches; the edited region jitters or appears static against a moving background.\n3 General style is similar, but the inpainted background's lighting/colours clearly clash or are inconsistent across frames; noticeable temporal disharmony.\n4 Style is almost uniform; minor edge issues around the removed area or slight temporal instability (e.g., minor flicker) visible only on close inspection.\n5 Perfectly seamless; the removal is temporally stable and visually indistinguishable from a clean background.\n\nPhysical & Detail Coherence\n1 Key original elements are blocked by poor inpainting; the background is heavily distorted or hallucinates incorrect structures; motion is completely wrong (e.g., a static patch in a moving scene).\n2 The inpainted background visibly shifts, jitters, or is poorly reconstructed over time, failing to match the original scene's motion.\n3 Background reconstruction is mostly correct and consistent; remaining flaws are small and acceptable; background changes are localized and stable.\n4 No loss of original detail around the removed area; background reconstruction is clean, stable, and respects the scene's geometry and motion.\n5 The background is essentially untouched and stable; the inpainted area perfectly matches the surrounding content's motion, texture, and detail over time.\n\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual & Temporal Naturalness: A number from 1 to 5.\nPhysical & Detail Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:"""

Local_Add = """\nYou are a data rater specializing in grading video object addition editing. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the edit quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 No edit performed, the video is corrupted, or the edit is completely wrong.\n2 Wrong object/class added, or target only partially added, or an unrelated object is also added.\n3 Correct object added, but with significant errors: key attributes (e.g., position, colour, count, size) are wrong.\n4 The correct object is added with main attributes correct; only minor details are off (e.g., slight colour mismatch, minor position error).\n5 Perfect: All and only the requested objects are added as instructed; every other element is untouched.\n\nVisual & Temporal Naturalness\n1 Video is badly broken, full of artefacts, or shows severe flickering/jittering throughout.\n2 Obvious paste marks; style, resolution, or palette of the added object strongly mismatches; the added region jitters or appears static against a moving background.\n3 General style is similar, but lighting/colours on the added object clearly clash or are inconsistent across frames; noticeable temporal disharmony.\n4 Style is almost uniform; minor edge issues around the added object or slight temporal instability (e.g., minor flicker) visible only on close inspection.\n5 Perfectly seamless; the edit is temporally stable and visually indistinguishable from the original video's content and motion.\n\nPhysical & Detail Coherence\n1 Severe physical errors (e.g., the added object floats, has wrong perspective/lighting); key original elements are blocked; motion of the added object is completely wrong.\n2 Contact with surfaces, occlusion by other objects, or motion of the added object is handled poorly.\n3 Lighting, perspective, and motion of the added object are mostly correct and consistent with the scene; remaining flaws are small and acceptable.\n4 Shadows, reflections, and material response from the added object are believable and move correctly with the scene; no loss of original detail.\n5 Edit enhances overall realism: the added object has precise highlights, shadows, and motion effects that are temporally coherent and perfectly integrated.\n\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual & Temporal Naturalness: A number from 1 to 5.\nPhysical & Detail Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:"""

Subtitle_Edit = """\nYou are a data rater specializing in grading instruction-following subtitle edits. You will be given two videos (before and after editing) and the corresponding editing instructions. Your task is to evaluate the subtitle edit on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 Target subtitle not added/removed/replaced, or wrong subtitle affected.\n2 Right action (add/remove/replace) but with incorrect content; only part of the edit is done; other subtitles are also altered.\n3 Mainly correct action and content, yet with significant spelling/grammar errors, or minor unintended edits to other subtitles.\n4 Correct action performed on the right subtitle; content is correct with only minor inaccuracies (e.g., small typos, punctuation errors).\n5 Exactly and only the requested subtitle(s) are added/removed/replaced; content matches the prompt perfectly; zero unintended edits.\n\nSubtitle Attribute Fidelity\n1 Completely fails to follow specified attributes (e.g., wrong position, wrong color). If attributes are not specified, the chosen ones make the subtitle unreadable or are extremely disruptive.\n2 Major deviation from specified attributes (e.g., requested bottom, placed on top). If not specified, chosen attributes are clearly wrong and distracting (e.g., obscures key visuals).\n3 Follows the general direction of specified attributes but with significant errors (e.g., correct side but wrong exact position). If not specified, chosen attributes are acceptable but noticeably inconsistent.\n4 Follows specified attributes with only minor inaccuracies (e.g., slightly off-center, minor deviation in font/color). If not specified, chosen attributes are highly appropriate with only minor flaws.\n5 All specified attributes (position, font, color, etc.) are matched perfectly. If attributes are not specified, the chosen ones are perfectly consistent with existing subtitles or professional standards.\n\nIntegrity of Unedited Content\n1 Massive collateral damage: background video is heavily corrupted/glitched, or other non-target subtitles are wrongly deleted/altered.\n2 Noticeable collateral damage: visible artifacts, distortion, or color shifts in the background video; other subtitles are visibly affected.\n3 Minor unintended effects: slight and localized visual artifacts in the background, or minor, non-critical changes to adjacent subtitles' appearance/timing.\n4 Almost perfect preservation: only extremely subtle artifacts in the video frame, visible only upon close inspection; all other subtitles are untouched.\n5 Perfect preservation: the edit is perfectly isolated; the background video and all other subtitles remain 100% identical to the original, with zero unintended changes.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nSubtitle Attribute Fidelity: A number from 1 to 5.\nIntegrity of Unedited Content: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the videos before and after editing:\n"""

EDIT_TYPE_TO_PROMPT = {
    "global_style": Global_Style,
    "creative_edit": Creative_Edit,
    "camera_edit": Camera_Edit,
    "local_change": Local_Change,
    "background_change": Background_Change,
    "local_remove": Local_Remove,
    "local_add": Local_Add,
    "subtitle_edit": Subtitle_Edit,
}


def generate_openve_csv(dataset, eval_root: str, accelerator):
    """Collect generated video paths and metadata into per-dataset OpenVE CSVs (distributed)."""
    fieldnames = ["edited_type", "prompt", "original_video", "edited_result_path"]

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

            rows_by_dataset[dataset_name].append({
                "edited_type": data_info.get("edited_type", ""),
                "prompt": data_info.get("instruction", data_info.get("prompt", "")),
                "original_video": data_info.get(
                    "source_video_path", data_info.get("original_video", "")
                ),
                "edited_result_path": output_video_path,
            })

    # Write per-rank temporary CSVs
    for dataset_name, rows in rows_by_dataset.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        temp_csv_path = os.path.join(
            metrics_dir, f"openve_input_rank{accelerator.process_index}.csv"
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
                    metrics_dir, f"openve_input_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if merged_rows:
                csv_path = os.path.join(metrics_dir, "openve_input.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(merged_rows)
                logging.info("Wrote %d rows to %s", len(merged_rows), csv_path)
            else:
                logging.warning(
                    "No valid outputs found for dataset %s", dataset_name
                )

    accelerator.wait_for_everyone()


def _extract_scores_and_average(response_text: str):
    """Extract numeric scores from Gemini response and return their average."""
    pattern = r":\s*(\d+\.?\d*)"
    matches = re.findall(pattern, response_text)
    scores = []
    for match in matches:
        try:
            scores.append(float(match))
        except ValueError:
            continue
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def _call_gemini(
    original_video_path: str,
    edited_video_path: str,
    prompt: str,
    gemini_url: str,
    gemini_headers: dict,
    max_retries: int = 5,
):
    """Call the Gemini model via the generateContent API with two inline videos."""
    for attempt in range(max_retries):
        try:
            parts = [{"text": prompt.strip()}]

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
                    for part in candidates[0].get("content", {}).get("parts", []):
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


def _score_single_sample(
    edited_type: str,
    prompt: str,
    original_video: str,
    edited_result_path: str,
    original_video_root: str,
    gemini_url: str,
    gemini_headers: dict,
):
    """Score a single edited video sample via Gemini."""
    base_row = {
        "edited_type": edited_type,
        "prompt": prompt,
        "original_video": original_video,
        "edited_result_path": edited_result_path,
    }

    # Resolve original video path
    resolved_original = original_video
    if original_video_root and not os.path.isabs(original_video):
        resolved_original = os.path.join(original_video_root, original_video)

    if not os.path.exists(edited_result_path):
        logging.warning("Edited video not found: %s", edited_result_path)
        return {
            **base_row,
            "results": f"ERROR: Video not found: {edited_result_path}",
            "average": "ERROR",
        }

    if not os.path.exists(resolved_original):
        logging.warning("Original video not found: %s", resolved_original)
        return {
            **base_row,
            "results": f"ERROR: Video not found: {resolved_original}",
            "average": "ERROR",
        }

    system_prompt = EDIT_TYPE_TO_PROMPT.get(edited_type)
    if system_prompt is None:
        error_msg = f"Unknown edit type: {edited_type}"
        logging.error(error_msg)
        return {**base_row, "results": f"ERROR: {error_msg}", "average": "ERROR"}

    full_prompt = system_prompt.replace("<edit_prompt>", prompt)

    try:
        response = _call_gemini(
            resolved_original,
            edited_result_path,
            full_prompt,
            gemini_url,
            gemini_headers,
        )
        formatted_response = response.replace("\n", "\\n")
        average_score = _extract_scores_and_average(response)
        return {
            **base_row,
            "results": formatted_response,
            "average": average_score,
        }
    except Exception as exc:
        logging.error("Error scoring sample: %s", exc)
        return {**base_row, "results": f"ERROR: {exc}", "average": "ERROR"}


def _aggregate_scores(all_scores, scores_by_type):
    """Compute per-type and overall averages (only valid 1-5 scores)."""
    type_averages = {}
    breakdown = {}
    for edited_type, scores in scores_by_type.items():
        valid = [s for s in scores if 1 <= s <= 5]
        type_averages[edited_type] = (
            round(sum(valid) / len(valid), 2) if valid else None
        )
        breakdown[edited_type] = {
            "count": len(valid),
            "average": type_averages[edited_type],
            "original_count": len(scores),
            "invalid_count": len(scores) - len(valid),
        }

    valid_all = [s for s in all_scores if 1 <= s <= 5]
    overall_average = (
        round(sum(valid_all) / len(valid_all), 2) if valid_all else None
    )

    return {
        "overall_average": overall_average,
        "type_averages": type_averages,
        "total_processed": len(all_scores),
        "total_valid_scores": len(valid_all),
        "breakdown_by_type": breakdown,
    }


def evaluate_on_openve(dataset, eval_root: str, accelerator, config):
    """Full OpenVE-Bench evaluation pipeline.

    Requires config fields under ``config.model.evaluation.gemini``:
        - api_key: Bearer token for Authorization header
        - base_url: Gemini API base URL
        - model: Model identifier (default: gemini-2.5-pro-06-17)
    """
    # Step 1: Generate evaluation CSV
    generate_openve_csv(
        dataset=dataset, eval_root=eval_root, accelerator=accelerator
    )

    # Read Gemini config
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

    # Resolve source data root from dataset config
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

    # Step 2: Distributed Gemini scoring
    scored_fieldnames = [
        "edited_type", "prompt", "original_video",
        "edited_result_path", "results", "average",
    ]

    for dataset_name in dataset.dataset_names:
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        input_csv = os.path.join(metrics_dir, "openve_input.csv")

        if not os.path.exists(input_csv):
            logging.warning(
                "No openve_input.csv found for dataset %s, skipping.",
                dataset_name,
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
                    "[Rank %d] Scoring row %d/%d (type=%s) ...",
                    accelerator.process_index,
                    row_idx + 1,
                    len(all_rows),
                    row.get("edited_type", ""),
                )
                scored_row = _score_single_sample(
                    edited_type=row.get("edited_type", ""),
                    prompt=row.get("prompt", ""),
                    original_video=row.get("original_video", ""),
                    edited_result_path=row.get("edited_result_path", ""),
                    original_video_root=original_video_root,
                    gemini_url=gemini_url,
                    gemini_headers=gemini_headers,
                )
                temp_rows.append(scored_row)

        temp_csv_path = os.path.join(
            metrics_dir, f"openve_scored_rank{accelerator.process_index}.csv"
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
                    metrics_dir, f"openve_scored_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as f:
                        merged_rows.extend(list(csv.DictReader(f)))
                    os.remove(temp_path)

            if not merged_rows:
                continue

            output_csv = os.path.join(metrics_dir, "openve_scored.csv")
            with open(output_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=scored_fieldnames)
                writer.writeheader()
                writer.writerows(merged_rows)

            all_scores = []
            all_scores_by_type = defaultdict(list)
            for row in merged_rows:
                avg = row.get("average", "ERROR")
                if avg != "ERROR":
                    try:
                        score_val = float(avg)
                        all_scores.append(score_val)
                        all_scores_by_type[row.get("edited_type", "")].append(
                            score_val
                        )
                    except (ValueError, TypeError):
                        pass

            stats = _aggregate_scores(all_scores, all_scores_by_type)

            summary_path = os.path.join(metrics_dir, "openve.json")
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            logging.info(
                "OpenVE-Bench [%s]: overall=%.2f, per-type=%s",
                dataset_name,
                stats.get("overall_average", 0) or 0,
                stats.get("type_averages", {}),
            )

    accelerator.wait_for_everyone()
