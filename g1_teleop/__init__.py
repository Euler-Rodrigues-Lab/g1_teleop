# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""g1_teleop — Unitree G1 teleoperation on the geo_kin_core interface."""

from pathlib import Path

__version__ = "0.1.0"

ASSETS_DIR = Path(__file__).parent / "assets"
SPECS_DIR = ASSETS_DIR / "specs"

# MJCF variants shipped with the package (meshdir="meshes" resolves in-tree).
XML_POSITION_CTRL = ASSETS_DIR / "g1_29dof_position_ctrl.xml"
XML_POSITION_CTRL_DANCE = ASSETS_DIR / "g1_29dof_position_ctrl_dance.xml"
XML_POSITION_CTRL_DANCE_W_HANDS = ASSETS_DIR / "g1_29dof_position_ctrl_dance_w_hands.xml"
URDF_29DOF = ASSETS_DIR / "g1_29dof.urdf"

# Inspire-hand-mounted G1, byte-identical copy from the monolith and
# signature-locked (the inspire spec npz artifacts embed its sha256, checked
# by geo_kin_core.spec.verify_signature). Its embedded meshdir points at the
# monolith layout, so load it through g1_teleop.mjcf.load_mjcf, which repoints
# the meshdir at assets/meshes in-memory; every mesh it references ships
# there. The matching FTP URDF is copied alongside for provenance/IK use.
XML_INSPIRE_MOUNTED = (
    ASSETS_DIR / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.xml"
)
URDF_INSPIRE_MOUNTED = (
    ASSETS_DIR / "g1_with_inspire_hand" / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf"
)
