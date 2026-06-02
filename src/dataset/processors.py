import os


instruction_prefixes = {
    "image_generation": "Generate an image: ",
    "image_generation_withref": "Generate an image with reference images: ",
    "image_reconstruction": "Reconstruct this image: ",
    "image_edit": "Edit this image: ",
    "video_generation": "Generate a video: ",
    "video_generation_withref": "Generate a video with reference images: ",
    "video_reconstruction": "Reconstruct this video: ",
    "video_edit": "Edit this video: ",
    "video_edit_withref": "Edit this video with reference images: "
}


# Image Generation
def process_t2i_data(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": instruction_prefixes["image_generation"] + data_info["caption"], 
            "is_target": False
        },
        {
            "type": "image", 
            "path": os.path.join(dataset_info["data_root"], data_info["image_path"]), 
            "rel_path": data_info["image_path"],
            "is_target": True,
            "need_resize": True,
        }
    ]
    return segments

def process_t2i_data_withref(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": instruction_prefixes["image_generation_withref"] + data_info["instruction"], 
            "is_target": False
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.append(
        {
            "type": "image", 
            "path": os.path.join(dataset_info["data_root"], data_info["target_image_path"]), 
            "rel_path": data_info["target_image_path"],
            "is_target": True,
            "need_resize": True,
        }
    )

    return segments

def process_t2i_data_withref_wo_prefix(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": data_info["instruction"], 
            "is_target": False
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.append(
        {
            "type": "image", 
            "path": os.path.join(dataset_info["data_root"], data_info["target_image_path"]), 
            "rel_path": data_info["target_image_path"],
            "is_target": True,
            "need_resize": True,
        }
    )

    return segments

def process_image_edit_data_wo_prefix_source_as_ref(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": data_info["instruction"], 
            "is_target": False
        },
    ]
    segments.append(
        {
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], data_info["source_image_path"]),
            "rel_path": data_info["source_image_path"],
            "is_target": False,
            "need_resize": True,
        }
    )
    segments.append(
        {
            "type": "image", 
            "path": os.path.join(dataset_info["data_root"], data_info["target_image_path"]), 
            "rel_path": data_info["target_image_path"],
            "is_target": True,
            "need_resize": True,
        }
    )

    return segments

def process_image_reconstruction_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["image_reconstruction"],
            "is_target": False
        },
        {
            "type": "source_image",
            "path": os.path.join(dataset_info["data_root"], data_info["image_path"]),
            "rel_path": data_info["image_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], data_info["image_path"]),
            "rel_path": data_info["image_path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

def process_image_edit_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["image_edit"],
            "is_target": False
        },
        {
            "type": "source_image",
            "path": os.path.join(dataset_info["data_root"], data_info["source_image_path"]),
            "rel_path": data_info["source_image_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        },
        {
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], data_info["target_image_path"]),
            "rel_path": data_info["target_image_path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

def process_image_edit_data_wo_prefix(dataset_info, data_info):
    segments = [
        {
            "type": "source_image",
            "path": os.path.join(dataset_info["data_root"], data_info["source_image_path"]),
            "rel_path": data_info["source_image_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        },
        {
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], data_info["target_image_path"]),
            "rel_path": data_info["target_image_path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

# Video Generation
def process_t2v_data(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": instruction_prefixes["video_generation"] + data_info["text"], 
            "is_target": False
        },
        {
            "type": "video", 
            "path": os.path.join(dataset_info["data_root"], data_info["path"]), 
            "rel_path": data_info["path"],
            "is_target": True,
            "need_resize": True,
        }
    ]

    return segments

def process_t2v_data_withref(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": instruction_prefixes["video_generation_withref"] + data_info["instruction"], 
            "is_target": False
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.append(
        {
            "type": "video", 
            "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]), 
            "rel_path": data_info["target_video_path"],
            "is_target": True,
            "need_resize": True,
        }
    )

    return segments

def process_t2v_data_withref_wo_prefix(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": data_info["instruction"], 
            "is_target": False
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.append(
        {
            "type": "video", 
            "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]), 
            "rel_path": data_info["target_video_path"],
            "is_target": True,
            "need_resize": True,
        }
    )

    return segments

def process_video_reconstruction_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_reconstruction"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["path"]),
            "rel_path": data_info["path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "video",
            "path": os.path.join(dataset_info["data_root"], data_info["path"]),
            "rel_path": data_info["path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

def process_video_edit_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_edit"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        },
        {
            "type": "video",
            "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]),
            "rel_path": data_info["target_video_path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

def process_video_edit_data_wo_prefix(dataset_info, data_info):
    segments = [
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        },
        {
            "type": "video",
            "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]),
            "rel_path": data_info["target_video_path"],
            "is_target": True,
            "need_resize": True,
        },
    ]
    return segments

def process_video_edit_data_withref(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_edit_withref"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.extend(
        [
            {
                "type": "text",
                "content": data_info["instruction"],
                "is_target": False
            },
            {
                "type": "video",
                "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]),
                "rel_path": data_info["target_video_path"],
                "is_target": True,
                "need_resize": True,
            },
        ]
    )
    return segments

def process_video_edit_data_withref_wo_prefix(dataset_info, data_info):
    segments = [
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
    ]
    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })
    segments.extend(
        [
            {
                "type": "text",
                "content": data_info["instruction"],
                "is_target": False
            },
            {
                "type": "video",
                "path": os.path.join(dataset_info["data_root"], data_info["target_video_path"]),
                "rel_path": data_info["target_video_path"],
                "is_target": True,
                "need_resize": True,
            },
        ]
    )
    return segments

# Evaluation (No target)
def process_only_text_data(dataset_info, data_info):
    segments = [
        {
            "type": "text", 
            "content": instruction_prefixes["image_generation"] + data_info["caption"],
            "save_path": data_info["save_path"],
            "is_target": False
        }
    ]
    return segments

def process_only_text_for_video_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_generation"] + data_info["caption"],
            "save_path": data_info["save_path"],
            "is_target": False
        }
    ]
    return segments

def process_only_source_image_data(dataset_info, data_info):
    data_index = data_info.get("_data_index", 0)
    prefix = f"{data_index:04d}"
    src_dir = os.path.dirname(data_info["source_image_path"])
    src_name = os.path.basename(data_info["source_image_path"])
    save_path = os.path.join(src_dir, f"{prefix}_{src_name}")

    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["image_edit"],
            "is_target": False
        },
        {
            "type": "source_image",
            "path": os.path.join(dataset_info["data_root"], data_info["source_image_path"]),
            "rel_path": data_info["source_image_path"],
            "save_path": save_path,
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        }
    ]
    return segments

def process_only_source_video_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_edit"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "save_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
        {
            "type": "text",
            "content": data_info["instruction"],
            "is_target": False
        }
    ]
    return segments

def process_source_video_with_ref_image_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_edit"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "save_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        },
    ]

    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })

    segments.append({
        "type": "text",
        "content": data_info["instruction"],
        "is_target": False
    })
    return segments


def process_refvie_data(dataset_info, data_info):
    """Like process_source_video_with_ref_image_data but prefixes save_path
    with a four-digit data_index (e.g. ``0042_videos/xxx.mp4``) so that
    entries sharing the same source video get distinct output paths."""
    data_index = data_info.get("_data_index", 0)
    prefix = f"{data_index:04d}"
    src_dir = os.path.dirname(data_info["source_video_path"])
    src_name = os.path.basename(data_info["source_video_path"])
    save_path = os.path.join(src_dir, f"{prefix}_{src_name}")

    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_edit"],
            "is_target": False
        },
        {
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "save_path": save_path,
            "is_target": False,
            "need_resize": True,
        },
    ]

    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })

    segments.append({
        "type": "text",
        "content": data_info["instruction"],
        "is_target": False
    })
    return segments

def process_intelligent_vbench_si2v_data(dataset_info, data_info):
    segments = [
        {
            "type": "text",
            "content": instruction_prefixes["video_generation_withref"],
            "save_path": data_info["save_path"],
            "is_target": False
        }
    ]

    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })

    segments.append({
        "type": "text",
        "content": data_info["instruction"],
        "is_target": False
    })
    return segments

def process_fashionvideobench_data(dataset_info, data_info):
    segments = []
    
    if len(data_info["source_video_path"]) > 0:
        segments.append({
            "type": "source_video",
            "path": os.path.join(dataset_info["data_root"], data_info["source_video_path"]),
            "rel_path": data_info["source_video_path"],
            "is_target": False,
            "need_resize": True,
        })

    for ref_image_path in data_info["reference_image_paths"]:
        segments.append({
            "type": "image",
            "path": os.path.join(dataset_info["data_root"], ref_image_path),
            "rel_path": ref_image_path,
            "is_target": False,
            "need_resize": True,
        })

    segments.append({
        "type": "text",
        "content": data_info["instruction"],
        "save_path": data_info["save_path"],
        "is_target": False
    })
    return segments