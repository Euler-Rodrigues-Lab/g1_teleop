"""Supply a valid G1 model to the optional public solver fallback."""
from pathlib import Path
from tempfile import TemporaryDirectory
import mujoco
from geo_kin_core.session import resolve_session
from g1_teleop import SPECS_DIR


def resolve_g1_session(model, **config):
    # Reference leg/arm chains use the specs shipped with this robot package.
    # The Rust wheel embeds these specs; MINK accepts and ignores this option.
    config.setdefault("spec_dir", SPECS_DIR)
    # The loader has already repaired model paths/keyframes. Serialize that model
    # so MINK sees the same geometry; licensed/reference backends ignore model_xml.
    with TemporaryDirectory(prefix="g1-model-") as directory:
        path = Path(directory) / "model.xml"
        mujoco.mj_saveLastXML(str(path), model)
        return resolve_session(robot="g1", model_xml=path, **config)
