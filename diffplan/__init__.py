# Intentionally empty: IsaacGym must be imported before torch, so `import diffplan.tasks.furniture`
# (which imports isaacgym first) has to stay safe and `import diffplan.dataset` must not drag in a
# simulator. Do not import torch, or any submodule, here.
