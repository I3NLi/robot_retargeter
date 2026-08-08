# Kengo local asset

This directory is the local asset slot used by `config/robot/kengo.yaml`.
The source bundle's README says **"Do Not distribute"**, so the raw MJCF and
STL files are intentionally ignored by Git and must not be pushed to the
public CCRP upstream.

The working copy uses the supplied `kengo_with_fist` model because that is the
23-DoF embodiment used by the local Kengo training and deployment stacks. The
required local layout is:

```text
kengo_description/
  meshes/*.STL
  mjcf/kengo.xml
```

`mjcf/kengo.xml` is the supplied fist MJCF with only the retargeting-specific
changes needed to match the G1 contract:

- no embedded floor, skybox, or scene light;
- fixed `hips_sphere`, `neck_sphere`, and `head_sphere` semantic bodies;
- fixed left/right heel (`*_foot_end_link`) and toe (`*_toe_link`) bodies;
- the original 23 joint names and qpos order remain unchanged.

To reproduce the local MJCF from the supplied `kengo_with_fist` bundle, copy
its STL files into `meshes/`, copy its XML to `mjcf/kengo.xml`, remove only
the embedded scene elements, and add these invisible fixed bodies (positions
are relative to their parent body):

```text
torso_link / neck_sphere:                    0          0          0.25767
neck_sphere / head_sphere:                   0.006      0          0.19633
pelvis_link / hips_sphere:                   0.043623   0         -0.041683
left_ankle_roll_link / left_foot_end_link:  -0.05       0         -0.035
left_ankle_roll_link / left_toe_link:        0.13       0         -0.035
right_ankle_roll_link / right_foot_end_link:-0.05       0         -0.035
right_ankle_roll_link / right_toe_link:      0.13       0         -0.035
```

Kengo output CSV rows contain 30 values:

```text
root_xyz(3) + root_quat_xyzw(4) + joint_pos(23)
```

The joint columns follow the MJCF/controller order: left arm (5), right arm
(5), waist yaw (1), left leg (6), right leg (6). Kengo's floating root is
`torso_link` and its pelvis is the child of `waist_yaw_joint`; this differs
from G1 and must not be reordered to imitate G1's numeric qpos layout.
