# Sample human motion

`picking_up_mustard.npz` — the full 14 s take (843 frames @ 60 Hz, ~1.3 MB) of
recorded human motion, stored as a **`geo_kin_core` frame stream** (schema
`geo_kin_core.frames/1`; see `geo_kin_core/frames.py`).

It exists so the offline replay demo and its tests run on a clean checkout —
no capture device, no hardware, and no additional source checkout:

```bash
python -m g1_teleop.demos.replay_offline          # uses this file by default
```

It doubles as the **data template** for new input devices: a frame stream is
just a recorded sequence of `RetargetFrame`s, so whatever a device adapter
produces should look like this when saved. Inspect one with:

```python
from geo_kin_core.frames import load_frames
stream = load_frames("g1_teleop/assets/sample_motion/picking_up_mustard.npz")
print(len(stream), stream.fps, stream.source)
frame = stream[0]        # a live-shaped RetargetFrame
```

Provenance: transcoded from the author's MIT-licensed `picking_up_mustard.csv`, via
`python -m g1_teleop.scripts.transcode_recording` (pass `--start` / `--duration`
to cut a shorter clip). Replaying the stream produces bit-identical solver
output to replaying the original CSV.
