# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""MJCF loading helpers.

Some MJCF variants copied byte-identically from the monolith are not directly
compilable from their committed location, and all of them are signature-locked
(their sha256 is embedded in the committed spec npz artifacts), so the files
cannot be edited. :func:`load_mjcf` repairs them in-memory when (and only
when) the direct load fails:

* The dance variants (``g1_29dof_position_ctrl_dance*.xml``) carry a stale
  ``stand`` keyframe whose ``ctrl`` row has 6 extra leading entries left over
  from an older mocap-ctrl layout. MuJoCo >= 3 rejects the size mismatch at
  compile time, so the ``<keyframe>`` block is dropped; nothing in the teleop
  pipeline uses the keyframes.
* The inspire-mounted variant
  (``g1_with_inspire_hand/g1_29dof_rev_1_0_with_inspire_hand_FTP.xml``)
  declares the monolith-layout
  ``meshdir="../../g1_full_body_kinematic/g1/meshes/"``. Every mesh it
  references ships in ``assets/meshes/``, so when the declared meshdir does
  not exist on disk it is repointed there.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

# In-tree mesh set shared by all monolith MJCF variants; used when a
# signature-locked XML declares a meshdir that only exists in the monolith
# layout.
FALLBACK_MESHDIR = Path(__file__).parent / "assets" / "meshes"


def load_mjcf(xml_path) -> mujoco.MjModel:
    """Load an MJCF, repairing known-broken keyframes/meshdirs without touching the file."""
    xml_path = Path(xml_path)
    try:
        return mujoco.MjModel.from_xml_path(str(xml_path))
    except ValueError as e:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        compiler = root.find("compiler")
        declared_meshdir = None
        if compiler is not None:
            declared_meshdir = (xml_path.parent / compiler.get("meshdir", "")).resolve()

        keyframe_broken = "keyframe" in str(e).lower()
        meshdir_broken = declared_meshdir is not None and not declared_meshdir.is_dir()
        if not (keyframe_broken or meshdir_broken):
            raise

    if keyframe_broken:
        for keyframe in root.findall("keyframe"):
            root.remove(keyframe)

    # from_xml_string has no base directory: make meshdir absolute (pointing
    # a monolith-layout meshdir at the in-tree mesh set).
    if compiler is not None:
        compiler.set("meshdir", str(declared_meshdir if not meshdir_broken
                                    else FALLBACK_MESHDIR))

    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
