# Copyright (c) 2026 Chuizheng Kong. Licensed under the MIT License.
"""Make the repo-root g1_teleop package importable without installation."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
