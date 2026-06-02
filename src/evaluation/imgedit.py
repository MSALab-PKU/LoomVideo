"""
ImgEdit-Bench evaluation using GPT-4o API for image editing quality assessment.

Modified from ImgEdit(https://github.com/PKU-YuanGroup/ImgEdit/tree/main/Benchmark/Basic)
"""

import os
import csv
import json
import base64
import logging
from collections import defaultdict

from openai import OpenAI

# ---------------------------------------------------------------------------
# Evaluation prompts (one per edit type)
# ---------------------------------------------------------------------------

REPLACE_PROMPT = """\nYou are a data rater specializing in grading image replacement edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the replacement editing effect on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1 Target not replaced, or an unrelated object/part of the image edited.\n2 Only part of the target replaced, or wrong class/description used.\n3 Target largely replaced but other objects altered, remnants visible, or count/position clearly wrong.\n4 Correct object fully replaced; only minor attribute errors (colour, size, etc.).\n5 Perfect replacement: all and only the specified objects replaced; new objects\u2019 class, number, position, scale, pose, and detail exactly match the prompt.\n\nVisual Naturalness\n1 Image heavily broken or new object deformed / full of artefacts.\n2 Obvious seams/edges; strong mismatch in resolution or colour; background not restored.\n3 Basic style similar, but lighting or palette clashes; fuzzy edges, noise noticeable.\n4 Style almost uniform; tiny artefacts visible only when zoomed; casual viewers see no edit.\n5 Completely seamless; new objects blend fully with the scene, edit area undetectable.\n\nPhysical & Detail Integrity\n1 Floating or wrong-scale object, severe perspective/light errors; background heavily warped.\n2 Missing or incorrect shadows/reflections; poor occlusion; background visibly changed.\n3 Lighting, perspective and interactions mostly correct; minor acceptable flaws; background only locally affected.\n4 New object interacts realistically with the scene and preserves existing details.\n5 Physically flawless: perspective, shadows, reflections, and lighting are perfect; background untouched.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Naturalness: A number from 1 to 5.\nPhysical & Detail Integrity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

ADD_PROMPT = """\nYou are a data rater specializing in grading image addition edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the added object(s) on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1  Nothing added or the added content is corrupt.\n2  Added object is a wrong class or unrelated to the prompt.\n3  Correct class, but key attributes (position, colour, size, count, etc.) are wrong.\n4  Main attributes correct; only minor details off or 1-2 small features missing.\n5  Every stated attribute correct and scene logic reasonable; only microscopic flaws.\n\nVisual Naturalness\n1  Image badly broken or full of artefacts.\n2  Obvious paste marks; style, resolution, or palette strongly mismatch.\n3  General style similar, but lighting or colours clearly clash; noticeable disharmony.\n4  Style almost uniform; small edge issues visible only when zoomed.\n5  Perfect blend; no visible difference between added object and original image.\n\nPhysical & Detail Coherence\n1  Severe physical errors (floating, wrong perspective/light); key original elements blocked; background heavily distorted.\n2  Contact or occlusion handled poorly; minor background shifts, jaggies or noise; background visibly changed.\n3  Lighting, perspective, and contact mostly correct; remaining flaws small and acceptable; limited background change.\n4  Shadows, reflections, and material response believable; no loss of original detail; background changes are minute.\n5  Added object enhances overall realism: precise highlights, shadows, ambient effects; background essentially untouched.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Naturalness: A number from 1 to 5.\nPhysical & Detail Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

ADJUST_PROMPT = """\nYou are a data rater specializing in grading attribute alteration edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the attribute change on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1  Target not adjusted, wrong object touched, or geometry changed.\n2  Right object but wrong attribute value/direction; only part edited; other objects also altered; slight stretch/crop.\n3  Mainly correct object and attribute, yet large hue/brightness/texture error; minor collateral edits; visible jaggies/distortion.\n4  All requested objects adjusted, only their attributes changed; shape kept; small inaccuracy in colour, material or amount.\n5  Exactly and only the requested objects adjusted; colour, material, gloss etc. match the prompt perfectly; shape 100% intact; zero unintended edits.\n\nVisual Seamlessness\n1  Massive colour spill, mosaics or heavy noise; image nearly unusable.\n2  Clear smears/bleeding on edges; abrupt resolution or tone shift; highlights/shadows clipped; background gaps.\n3  Overall palette OK but local tone or grain conflicts; soft edges; noticeable disharmony.\n4  Style unified, transitions smooth; only slight edge artefacts visible when zoomed.\n5  No detectable edit traces; colours/materials fuse with scene lighting; edit area practically invisible.\n\nPhysical & Detail Fidelity\n1  Object floating, interpenetrating, or severe perspective/light mismatch; background badly warped.\n2  Missing shadows/highlights; wrong reflection direction; background visibly discoloured or distorted.\n3  Light, perspective and contact surface largely correct; minor acceptable flaws; background only locally affected.\n4  Adjusted material interacts believably with scene; shadows, highlights, reflections handled well; original details preserved.\n5  High physical realism: fine micro-highlights, diffuse bounce, subsurface effects present; overall scene realism improved.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Seamlessness: A number from 1 to 5.\nPhysical & Detail Fidelity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

REMOVE_PROMPT = """\nYou are a data rater specializing in grading object removal edits. You will be given two images (before and after editing) and the corresponding editing instructions. Your task is to evaluate the removal quality on a 5-point scale from three perspectives:\n\nPrompt Compliance\n1  Nothing removed, or an unrelated object edited.\n2  Target only partly removed, or a different instance/class deleted, or another object appears in the gap.\n3  Target mostly removed but extra objects also deleted, or fragments of the target remain.\n4  Only the specified objects removed, but a few tiny/background items deleted by mistake, or the count is wrong.\n5  Perfect: all and only the requested objects removed; every other element untouched.\n\nVisual Naturalness\n1  Image badly broken (large holes, strong artefacts).\n2  Clear erase marks; colour/resolution mismatch; background not restored.\n3  General look acceptable yet lighting/colour/style still clash; blur or noise visible.\n4  Style consistent; minor edge issues visible only when zoomed.\n5  Seamless: removal is virtually impossible to spot.\n\nPhysical & Detail Integrity\n1  Severe physical errors (floating items, wrong perspective/light); key scene elements damaged; background heavily warped.\n2  Large un-filled gaps or obvious background shifts.\n3  Lighting, perspective and contacts mostly correct; flaws small and tolerable; background adjusted locally.\n4  Background reconstruction clean; existing details preserved; only minute changes outside the removal area.\n5  Physically flawless and even enhances realism: accurate light/shadow/texture infill, high-quality micro-details.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nPrompt Compliance: A number from 1 to 5.\nVisual Naturalness: A number from 1 to 5.\nPhysical & Detail Integrity: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

STYLE_PROMPT = """\nYou are a data rater specializing in grading style transfer edits. You will be given an input image, a reference style, and the styled result. Your task is to evaluate the style transfer on a 5-point scale from three perspectives:\n\nStyle Fidelity\n1  Target style absent or clearly wrong.\n2  Style shows in a few areas only, or mixed with unrelated styles.\n3  Key traits (palette, brushwork, texture) present but patchy or inconsistent.\n4  Style reproduced across almost the whole image; only small local mismatches.\n5  Full, faithful transfer: colour, texture, brushwork, lighting all match the exemplar over the entire image.\n\nContent Preservation\n1  Major objects or layout lost/distorted; original scene barely recognisable.\n2  Main subject recognisable, but size, perspective or key parts clearly wrong/missing.\n3  Overall structure correct; some local warping or minor omissions.\n4  Nearly all geometry intact; only slight, non-distracting deformation.\n5  All objects and spatial relations kept; only stylistic, harmless distortion.\n\nRendering Quality\n1  Heavy noise, banding, pixel damage or blur; image unusable.\n2  Visible seams, aliasing, colour drift; low resolution or chaotic strokes.\n3  Moderate quality: local blur/noise/texture breaks, but generally acceptable.\n4  Sharp, coherent strokes; tiny artefacts visible only when zoomed.\n5  High resolution, no artefacts; strokes, textures and colour transitions look fully natural.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nStyle Fidelity: A number from 1 to 5.\nContent Preservation: A number from 1 to 5.\nRendering Quality: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the input, reference style, and styled output image:\n"""

ACTION_PROMPT = """\nYou are a data rater specializing in grading action or expression change edits. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the motion or expression change on a 5-point scale from three perspectives:\n\nAction / Expression Fidelity\n1  No visible change, or wrong action / expression.\n2  Partial or clearly incorrect pose; only some body parts change; expression direction wrong.\n3  Main idea present but details off (angle, side, intensity, missing gesture).\n4  Requested pose / expression achieved with just minor inaccuracy (small angular drift, timing nuance).\n5  Exact match to prompt: every limb, gesture, and facial muscle aligns with the described action.\n\nIdentity Preservation\n1  Person unrecognisable; face or body replaced.\n2  Strong drift: key facial features, hairstyle or clothing heavily altered.\n3  Mostly same identity; moderate changes in some features but still recognisable.\n4  Identity clearly the same; only subtle stylisation or lighting differences.\n5  Perfect preservation of face, hairstyle, skin tone, clothing and accessories.\n\nVisual & Anatomical Coherence\n1  Severe artifacts: broken or duplicated limbs, extreme distortion, heavy noise/blur.\n2  Noticeable cut-out halos, proportion errors, lighting or perspective clearly off.\n3  Generally plausible; minor joint or shading issues; small noise/blur acceptable.\n4  Clean render; anatomy, lighting, depth and edges consistent; flaws only on close inspection.\n5  Flawless realism or stylistic coherence; perfect anatomy, lighting, shadows and texture continuity.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nAction Fidelity: A number from 1 to 5.\nIdentity Preservation: A number from 1 to 5.\nVisual & Anatomical Coherence: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

EXTRACT_PROMPT = """\nYou are a data rater specializing in grading object cut-out quality. You will be given an image with the object extracted on a white background. Your task is to evaluate the cut-out accuracy on a 5-point scale from three perspectives:\n\nObject Selection & Identity\n1  Wrong object or multiple objects extracted.\n2  Correct class but only part of the object, or obvious intrusions from other items.\n3  Object largely correct yet small pieces missing / extra, identity still recognisable.\n4  Full object with clear identity; only tiny mis-crop (e.g., tip of antenna).\n5  Exact requested object, complete and unmistakably the same instance (ID).\n\nMask Precision & Background Purity\n1  Large background remnants, holes in mask, or non-white backdrop dominates.\n2  Noticeable jagged edges, colour fringes, grey/colour patches in white area.\n3  Acceptable mask; minor edge softness or faint halo visible on close look.\n4  Clean, smooth edges; white (#FFFFFF) background uniform, tiny artefacts only when zoomed.\n5  Crisp anti-aliased contour, zero spill or halo; backdrop perfectly pure white throughout.\n\nObject Integrity & Visual Quality\n1  Severe blur, compression, deformation, or missing parts; unusable.\n2  Moderate noise, colour shift, or slight warping; details clearly degraded.\n3  Overall intact with minor softness or noise; colours mostly preserved.\n4  Sharp detail, accurate colours; negligible artefacts.\n5  Pristine: high-resolution detail, true colours, no artefacts or distortion.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nObject Identity: A number from 1 to 5.\nMask Precision: A number from 1 to 5.\nVisual Quality: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow is the extracted object image:\n"""

BACKGROUND_PROMPT = """\nYou are a data rater specializing in grading background editing. You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the background change on a 5-point scale from three perspectives:\n\nInstruction Compliance\n1  No change, or background unrelated to prompt, or foreground also replaced/distorted.\n2  Background partly replaced or wrong style/content; foreground noticeably altered.\n3  Main background replaced but elements missing/extra, or faint spill onto subject edges.\n4  Requested background fully present; foreground intact except minute artefacts or small prompt mismatch (e.g. colour tone).\n5  Background exactly matches prompt (content, style, placement); all foreground pixels untouched.\n\nVisual Seamlessness (Edge & Texture Blend)\n1  Large tearing, posterisation, extreme blur/noise; edit area obvious at a glance.\n2  Clear cut-out halos, colour-resolution gap, or heavy smudge strokes.\n3  Blend acceptable but visible on closer look: slight edge blur, grain or palette shift.\n4  Nearly invisible seams; textures and sharpness aligned, only minor issues when zoomed in.\n5  Indistinguishable composite: edges, textures, resolution and colour grading perfectly continuous.\n\nPhysical Consistency (Lighting, Perspective, Depth)\n1  Severe mismatch: wrong horizon, conflicting light direction, floating subject, warped geometry.\n2  Noticeable but not extreme inconsistencies in light, shadows or scale; depth cues off.\n3  Overall believable; small errors in shadow length, perspective or ambient colour.\n4  Lighting, scale, depth, and camera angle well matched; only subtle discrepancies.\n5  Physically flawless: foreground and new background share coherent light, shadows, reflections, perspective and atmospheric depth, enhancing overall realism.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nInstruction Compliance: A number from 1 to 5.\nVisual Seamlessness: A number from 1 to 5.\nPhysical Consistency: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

COMPOSE_PROMPT = """\nYou are a data rater specializing in grading hybrid image edits (involving multiple operations on multiple objects). You will be given two images (before and after editing) and the editing instruction. Your task is to evaluate the overall editing quality on a 5-point scale from three perspectives:\n\nInstruction Compliance\n1  Neither object nor operations match the prompt; wrong items edited or shapes distorted.\n2  Only one object correctly edited, or both edited but with wrong/partial operations; collateral changes to other items.\n3  Both target objects touched, each with the requested operation broadly correct but missing details (e.g., wrong colour value, incomplete removal).\n4  Both objects receive the exact operations; tiny deviations in amount, position, or parameter. No unintended edits elsewhere.\n5  Perfect execution: each object fully reflects its specified operation, all other scene elements untouched.\n\nVisual Naturalness (Seamlessness)\n1  Large artefacts, obvious cut-outs, heavy blur/noise; edits conspicuous at a glance.\n2  Clear edge halos, colour or resolution mismatch, awkward scaling.\n3  Acceptable but visible on close look: slight edge softness, minor palette or focus shift.\n4  Edits blend smoothly; seams hard to spot, textures and sharpness largely consistent.\n5  Indistinguishable composite: colour grading, grain, resolution and style fully match the original image.\n\nPhysical Consistency & Fine Detail\n1  Severe lighting/perspective mismatch, missing or wrong shadows; objects appear floating or warped.\n2  Noticeable but tolerable inconsistencies in illumination, scale, or depth cues.\n3  Generally plausible; small errors in shadow length, reflection angle, or texture alignment.\n4  Lighting, perspective, and material response closely match; only subtle flaws visible when zoomed.\n5  Physically flawless: shadows, highlights, reflections, depth and texture perfectly integrated, enhancing overall realism.\nThe second and third score should no higher than first score!!!\n\nExample Response Format:\nBrief reasoning: A short explanation of the score based on the criteria above, no more than 20 words.\nInstruction Compliance: A number from 1 to 5.\nVisual Naturalness: A number from 1 to 5.\nPhysical Consistency & Fine Detail: A number from 1 to 5.\nediting instruction is : <edit_prompt>.\n\nBelow are the images before and after editing:\n"""

EDIT_TYPE_TO_PROMPT = {
    "replace": REPLACE_PROMPT,
    "add": ADD_PROMPT,
    "adjust": ADJUST_PROMPT,
    "remove": REMOVE_PROMPT,
    "style": STYLE_PROMPT,
    "action": ACTION_PROMPT,
    "extract": EXTRACT_PROMPT,
    "background": BACKGROUND_PROMPT,
    "compose": COMPOSE_PROMPT,
}


def _image_to_base64(image_path: str):
    """Read an image file and return its base64-encoded string, or None on error."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        logging.warning("File %s not found.", image_path)
        return None


def generate_imgedit_csv(dataset, eval_root: str, accelerator):
    """Collect generated image paths and metadata into per-dataset ImgEdit CSVs."""
    fieldnames = ["edit_type", "prompt", "original_image", "edited_result_path"]

    all_indices = list(range(len(dataset)))
    local_indices = all_indices[accelerator.process_index :: accelerator.num_processes]

    rows_by_dataset = defaultdict(list)
    for idx in local_indices:
        data = dataset[idx]
        if data is None:
            continue

        dataset_name = data["dataset_name"]
        for gen_relative_path in data["gen_paths"]:
            output_image_path = os.path.join(
                eval_root, dataset_name, "outputs", gen_relative_path
            )
            # Try common image extensions if the raw path doesn't exist
            if not os.path.exists(output_image_path):
                for ext in (".png", ".jpg", ".jpeg"):
                    candidate = os.path.splitext(output_image_path)[0] + ext
                    if os.path.exists(candidate):
                        output_image_path = candidate
                        break
                else:
                    continue

            json_path = os.path.splitext(output_image_path)[0] + ".json"
            if not os.path.exists(json_path):
                continue

            with open(json_path, "r", encoding="utf-8") as fh:
                data_info = json.load(fh)

            edit_type = data_info.get(
                "edit_type", data_info.get("edited_type", "")
            )
            prompt = data_info.get("instruction", data_info.get("prompt", ""))
            original_image = data_info.get(
                "source_image_path",
                data_info.get(
                    "original_image",
                    data_info.get("source_video_path", ""),
                ),
            )

            rows_by_dataset[dataset_name].append({
                "edit_type": edit_type,
                "prompt": prompt,
                "original_image": original_image,
                "edited_result_path": output_image_path,
            })

    # Write per-rank temporary CSVs
    for dataset_name, rows in rows_by_dataset.items():
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        temp_csv_path = os.path.join(
            metrics_dir, f"imgedit_input_rank{accelerator.process_index}.csv"
        )
        with open(temp_csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
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
                    metrics_dir, f"imgedit_input_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as fh:
                        merged_rows.extend(list(csv.DictReader(fh)))
                    os.remove(temp_path)

            if merged_rows:
                csv_path = os.path.join(metrics_dir, "imgedit_input.csv")
                with open(csv_path, "w", encoding="utf-8", newline="") as fh:
                    writer = csv.DictWriter(fh, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(merged_rows)
                logging.info(
                    "Wrote %d rows to %s", len(merged_rows), csv_path
                )
            else:
                logging.warning(
                    "No valid ImgEdit outputs for dataset %s", dataset_name
                )

    accelerator.wait_for_everyone()


def _extract_scores_and_average(response_text: str):
    """Extract ``dimension: score`` lines and return their average."""
    lines = response_text.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None


def _call_gpt(
    original_image_path: str,
    result_image_path: str,
    full_prompt: str,
    client: OpenAI,
    model: str,
):
    """Call GPT-4o with two base64-encoded images."""
    original_base64 = _image_to_base64(original_image_path)
    result_base64 = _image_to_base64(result_image_path)

    if not original_base64 or not result_base64:
        return "ERROR: Image conversion failed"

    response = client.chat.completions.create(
        model=model,
        stream=False,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{original_base64}"
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{result_base64}"
                        },
                    },
                ],
            }
        ],
    )

    return response.choices[0].message.content


def _score_single_sample(
    edit_type: str,
    prompt: str,
    original_image: str,
    edited_result_path: str,
    original_image_root: str,
    client: OpenAI,
    model: str,
):
    """Score a single edited image sample via GPT-4o."""
    base_row = {
        "edit_type": edit_type,
        "prompt": prompt,
        "original_image": original_image,
        "edited_result_path": edited_result_path,
    }

    # Resolve original image path relative to data root
    resolved_original = original_image
    if original_image_root and original_image and not os.path.isabs(original_image):
        resolved_original = os.path.join(original_image_root, original_image)

    if not os.path.exists(edited_result_path):
        logging.warning("Edited image not found: %s", edited_result_path)
        return {
            **base_row,
            "results": f"ERROR: Image not found: {edited_result_path}",
            "average": "ERROR",
        }

    if not os.path.exists(resolved_original):
        logging.warning("Original image not found: %s", resolved_original)
        return {
            **base_row,
            "results": f"ERROR: Image not found: {resolved_original}",
            "average": "ERROR",
        }

    system_prompt = EDIT_TYPE_TO_PROMPT.get(edit_type)
    if system_prompt is None:
        error_msg = f"Unknown edit type: {edit_type}"
        logging.error(error_msg)
        return {**base_row, "results": f"ERROR: {error_msg}", "average": "ERROR"}

    full_prompt = system_prompt.replace("<edit_prompt>", prompt)

    try:
        response_text = _call_gpt(
            resolved_original, edited_result_path, full_prompt, client, model
        )
        formatted_response = response_text.replace("\n", "\\n")
        average_score = _extract_scores_and_average(response_text)
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
    for edit_type, scores in scores_by_type.items():
        valid = [s for s in scores if 1 <= s <= 5]
        type_averages[edit_type] = (
            round(sum(valid) / len(valid), 2) if valid else None
        )
        breakdown[edit_type] = {
            "count": len(valid),
            "average": type_averages[edit_type],
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


def evaluate_on_imgedit(dataset, eval_root: str, accelerator, config):
    """
    Full ImgEdit-Bench evaluation pipeline.

    Requires ``config.model.evaluation.openai`` with fields:
        - api_key: OpenAI API key
        - base_url: OpenAI-compatible API base URL
        - model: Model name (default: gpt-4.1)
    """
    # Step 1: Generate evaluation CSV
    generate_imgedit_csv(
        dataset=dataset, eval_root=eval_root, accelerator=accelerator
    )

    # Read OpenAI config
    openai_config = getattr(config.model.evaluation, "openai", None)
    api_key = (
        getattr(openai_config, "api_key", None) if openai_config else None
    )
    base_url = (
        getattr(openai_config, "base_url", None) if openai_config else None
    )
    model = (
        getattr(openai_config, "model", None) if openai_config else None
    )

    # Fall back to environment variables
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = base_url or os.environ.get("OPENAI_BASE_URL", "")
    model = model or os.environ.get("OPENAI_MODEL", "gpt-4.1")

    if not api_key or not base_url:
        logging.error(
            "OpenAI API config missing. Set via config.model.evaluation.openai "
            "or env vars OPENAI_API_KEY / OPENAI_BASE_URL"
        )
        accelerator.wait_for_everyone()
        return

    client = OpenAI(api_key=api_key, base_url=base_url)

    # Resolve source data root from dataset config (local filesystem)
    dataset_name_to_data_root = {
        name: getattr(info, "data_root", "")
        for name, info in zip(dataset.dataset_names, dataset.dataset_list)
    }

    # Step 2: Distributed GPT-4o scoring
    scored_fieldnames = [
        "edit_type", "prompt", "original_image",
        "edited_result_path", "results", "average",
    ]

    for dataset_name in dataset.dataset_names:
        metrics_dir = os.path.join(eval_root, dataset_name, "metrics")
        input_csv = os.path.join(metrics_dir, "imgedit_input.csv")

        if not os.path.exists(input_csv):
            logging.warning(
                "No imgedit_input.csv found for dataset %s, skipping.",
                dataset_name,
            )
            continue

        original_image_root = dataset_name_to_data_root.get(dataset_name, "")

        with open(input_csv, "r", encoding="utf-8-sig") as fh:
            all_rows = list(csv.DictReader(fh))

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
                    row.get("edit_type", ""),
                )
                scored_row = _score_single_sample(
                    edit_type=row.get("edit_type", ""),
                    prompt=row.get("prompt", ""),
                    original_image=row.get("original_image", ""),
                    edited_result_path=row.get("edited_result_path", ""),
                    original_image_root=original_image_root,
                    client=client,
                    model=model,
                )
                temp_rows.append(scored_row)

        temp_csv_path = os.path.join(
            metrics_dir, f"imgedit_scored_rank{accelerator.process_index}.csv"
        )
        with open(temp_csv_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=scored_fieldnames)
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
                    metrics_dir, f"imgedit_scored_rank{proc_idx}.csv"
                )
                if os.path.exists(temp_path):
                    with open(temp_path, "r", encoding="utf-8") as fh:
                        merged_rows.extend(list(csv.DictReader(fh)))
                    os.remove(temp_path)

            if not merged_rows:
                continue

            output_csv = os.path.join(metrics_dir, "imgedit_scored.csv")
            with open(output_csv, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=scored_fieldnames)
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
                        all_scores_by_type[row.get("edit_type", "")].append(
                            score_val
                        )
                    except (ValueError, TypeError):
                        pass

            stats = _aggregate_scores(all_scores, all_scores_by_type)

            summary_path = os.path.join(metrics_dir, "imgedit.json")
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump(stats, fh, ensure_ascii=False, indent=2)

            logging.info(
                "ImgEdit-Bench [%s]: overall=%.2f, per-type=%s",
                dataset_name,
                stats.get("overall_average", 0) or 0,
                stats.get("type_averages", {}),
            )

    accelerator.wait_for_everyone()
