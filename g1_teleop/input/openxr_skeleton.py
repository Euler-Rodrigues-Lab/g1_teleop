# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""OpenXR full-body bone list -> the device-neutral skeleton a frame carries.

Ported from the monolith's ``projects/shared_scripts/mujoco_human_capsule.py``
hierarchy so the human overlay draws the same bones as the capture-side viewer:
spine chain, neck, head, shoulders, arms, wrists, every finger bone
(metacarpal -> tip), and legs down to the foot ball (toes).

The output is device-neutral on purpose — plain names, positions, and parent
indices — so :class:`geo_kin_core.viz.HumanCapsuleViz` needs no OpenXR enum,
and a recorded frame stream replays the skeleton with no device installed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

_FINGERS = ("Index", "Middle", "Ring", "Little")


def _hand_hierarchy(bone_id, side: str) -> dict:
    """Finger bone -> parent map for one hand (missing bones are skipped)."""
    prefix = f"FullBody_{side}Hand"
    wrist = getattr(bone_id, f"{prefix}Wrist", None)
    if wrist is None:
        return {}
    get = lambda name: getattr(bone_id, name, None)  # noqa: E731

    hierarchy = {}
    thumb = [get(f"{prefix}Thumb{part}") for part in
             ("Metacarpal", "Proximal", "Distal", "Tip")]
    parent = wrist
    for bone in thumb:
        if bone is not None:
            hierarchy[bone] = parent
            parent = bone

    for finger in _FINGERS:
        chain = [get(f"{prefix}{finger}{part}") for part in
                 ("Metacarpal", "Proximal", "Intermediate", "Distal", "Tip")]
        parent = wrist
        for bone in chain:
            if bone is not None:
                hierarchy[bone] = parent
                parent = bone
    return hierarchy


def full_body_hierarchy() -> dict:
    """Child -> parent map over ``FullBodyBoneId`` (empty if unavailable)."""
    try:
        from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId as B
    except ImportError:
        return {}

    hierarchy = {
        B.FullBody_SpineLower: B.FullBody_Hips,
        B.FullBody_SpineMiddle: B.FullBody_SpineLower,
        B.FullBody_SpineUpper: B.FullBody_SpineMiddle,
        B.FullBody_Neck: B.FullBody_SpineUpper,
        B.FullBody_Head: B.FullBody_Neck,
        B.FullBody_LeftShoulder: B.FullBody_SpineUpper,
        B.FullBody_LeftArmUpper: B.FullBody_LeftShoulder,
        B.FullBody_LeftArmLower: B.FullBody_LeftArmUpper,
        B.FullBody_LeftHandWrist: B.FullBody_LeftArmLower,
        B.FullBody_RightShoulder: B.FullBody_SpineUpper,
        B.FullBody_RightArmUpper: B.FullBody_RightShoulder,
        B.FullBody_RightArmLower: B.FullBody_RightArmUpper,
        B.FullBody_RightHandWrist: B.FullBody_RightArmLower,
        B.FullBody_LeftUpperLeg: B.FullBody_Hips,
        B.FullBody_LeftLowerLeg: B.FullBody_LeftUpperLeg,
        B.FullBody_LeftFootAnkle: B.FullBody_LeftLowerLeg,
        B.FullBody_LeftFootBall: B.FullBody_LeftFootAnkle,
        B.FullBody_RightUpperLeg: B.FullBody_Hips,
        B.FullBody_RightLowerLeg: B.FullBody_RightUpperLeg,
        B.FullBody_RightFootAnkle: B.FullBody_RightLowerLeg,
        B.FullBody_RightFootBall: B.FullBody_RightFootAnkle,
    }
    hierarchy.update(_hand_hierarchy(B, "Left"))
    hierarchy.update(_hand_hierarchy(B, "Right"))
    return hierarchy


def bones_to_skeleton(bones) -> Optional[dict]:
    """``[Bone]`` (or ``{id: Bone}``) -> ``{"positions", "names", "parents"}``.

    Only bones present in this capture AND in the hierarchy are emitted, so a
    body-only capture simply yields fewer bones. Returns None when the bone
    schema is unavailable or nothing matches.
    """
    if bones is None:
        return None
    hierarchy = full_body_hierarchy()
    if not hierarchy:
        return None
    bone_map = {b.id: b for b in bones} if isinstance(bones, (list, tuple)) else dict(bones)

    def raw_id(key):
        return key.value if hasattr(key, "value") else key

    ids = [i for i in bone_map if any(raw_id(c) == i for c in hierarchy)]
    ids += [raw_id(p) for c, p in hierarchy.items() if raw_id(c) in bone_map]
    ids = sorted(set(i for i in ids if i in bone_map))
    if not ids:
        return None

    index = {bone_id: k for k, bone_id in enumerate(ids)}
    parents = np.full(len(ids), -1, dtype=np.int64)
    for child, parent in hierarchy.items():
        c, p = raw_id(child), raw_id(parent)
        if c in index and p in index:
            parents[index[c]] = index[p]

    try:
        from xr_robot_teleop_server.schemas.openxr_skeletons import FullBodyBoneId as B
        name_of = lambda i: B(i).name  # noqa: E731
    except Exception:
        name_of = lambda i: f"bone_{i}"  # noqa: E731

    return {
        "positions": np.array([np.asarray(bone_map[i].position, dtype=float) for i in ids]),
        "names": tuple(name_of(i) for i in ids),
        "parents": parents,
    }
