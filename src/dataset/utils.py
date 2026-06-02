from typing import List, Optional

# Resolution buckets are in (width, height) format.
# Each bucket group is keyed by the approximate side length (e.g. "256", "640", "960").
# Within each group, multiple aspect ratios are provided.

IMAGE_RESOLUTION_BUCKETS = {
    "256": [
        (256, 256),    # 1:1
        (288, 224),    # 9:7
        (224, 288),    # 7:9
        (320, 224),    # ~3:2
        (224, 320),    # ~2:3
        (352, 192),    # ~16:9
        (192, 352),    # ~9:16
        (384, 160),    # ~21:9
        (160, 384),    # ~9:21
    ],
    # 480p
    "640": [
        (640, 640),    # 1:1
        (704, 544),    # 9:7
        (544, 704),    # 7:9
        (768, 512),    # ~3:2
        (512, 768),    # ~2:3
        (832, 480),    # ~16:9
        (480, 832),    # ~9:16
        (960, 416),    # ~21:9
        (416, 960),    # ~9:21
    ],
    # 720p
    "960": [
        (960, 960),    # 1:1
        (1088, 864),   # 9:7
        (864, 1088),   # 7:9
        (1152, 800),   # ~3:2
        (800, 1152),   # ~2:3
        (1280, 704),   # ~16:9
        (704, 1280),   # ~9:16
        (1472, 608),   # ~21:9
        (608, 1472),   # ~9:21
    ],
}

VIDEO_RESOLUTION_BUCKETS = {
    "256": [
        (256, 256),    # 1:1
        (288, 224),    # 9:7
        (224, 288),    # 7:9
        (320, 224),    # ~3:2
        (224, 320),    # ~2:3
        (352, 192),    # ~16:9
        (192, 352),    # ~9:16
        (384, 160),    # ~21:9
        (160, 384),    # ~9:21
    ],
    # 480p
    "640": [
        (640, 640),    # 1:1
        (704, 544),    # 9:7
        (544, 704),    # 7:9
        (768, 512),    # ~3:2
        (512, 768),    # ~2:3
        (832, 480),    # ~16:9
        (480, 832),    # ~9:16
        (960, 416),    # ~21:9
        (416, 960),    # ~9:21
    ],
    # 720p
    "960": [
        (960, 960),    # 1:1
        (1088, 864),   # 9:7
        (864, 1088),   # 7:9
        (1152, 800),   # ~3:2
        (800, 1152),   # ~2:3
        (1280, 704),   # ~16:9
        (704, 1280),   # ~9:16
        (960, 704),
        (704, 960),
        (1472, 608),   # ~21:9
        (608, 1472),   # ~9:21
    ],
}


def get_closest_resolution(
    width: int,
    height: int,
    res_buckets: dict,
    allowed_buckets: Optional[List[str]] = None,
):
    """
    Find the closest (height, width) from resolution buckets for a given image/video size.
    
    Args:
        width: Original width.
        height: Original height.
        res_buckets: Dict mapping bucket key (e.g. "640") to list of (w, h) tuples.
        allowed_buckets: If provided, only consider these bucket keys.

    Returns:
        (target_height, target_width) tuple.
    """
    original_area = width * height
    original_aspect_ratio = width / height

    closest_bucket_key = None
    min_area_diff = float("inf")

    available_keys = list(res_buckets.keys())
    if allowed_buckets is not None:
        available_keys = [k for k in available_keys if k in allowed_buckets]

    if not available_keys:
        raise ValueError(
            "No valid buckets found. Check your allowed_buckets and res_buckets."
        )

    # Step 1: find the bucket group with the closest total area
    for bucket_key in available_keys:
        bucket_area = int(bucket_key) ** 2
        area_diff = abs(bucket_area - original_area)
        if area_diff < min_area_diff:
            min_area_diff = area_diff
            closest_bucket_key = bucket_key

    # Step 2: within that group, find the closest aspect ratio
    closest_resolution = None
    min_aspect_diff = float("inf")

    for res_w, res_h in res_buckets[closest_bucket_key]:
        res_aspect_ratio = res_w / res_h
        aspect_diff = abs(res_aspect_ratio - original_aspect_ratio)
        if aspect_diff < min_aspect_diff:
            min_aspect_diff = aspect_diff
            closest_resolution = (res_h, res_w)

    return closest_resolution
