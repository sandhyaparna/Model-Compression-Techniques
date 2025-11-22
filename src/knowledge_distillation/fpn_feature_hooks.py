import torch.nn as nn


# ==============================================================================
# Hooks & helpers
# ==============================================================================
class FeatureHook:
    """Capture a dict[str->Tensor] of FPN features from backbone output."""

    def __init__(self):
        self.last = None

    def __call__(self, module, inputs, outputs):
        # outputs is typically an OrderedDict[str, Tensor]
        self.last = {k: v for k, v in outputs.items()}


class RPNHeadHook:
    """Capture RPN objectness logits per level (list[Tensor])."""

    def __init__(self):
        self.objectness = None

    def __call__(self, module, inputs, outputs):
        obj_logits, _ = outputs  # tuple(list[Tensor], list[Tensor])
        self.objectness = obj_logits


def _maybe_layer_1x1(in_ch, out_ch):
    if in_ch == out_ch:
        return nn.Identity()
    layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
    return layer


class FPNAdapter(nn.Module):
    """
    Align student FPN channels to teacher FPN channels by index (paired lists).
    """

    def __init__(self):
        super().__init__()
        self.adapters = nn.ModuleList()
        self.num_levels = None
        self._initialized = False

    def init_from_feature_shapes(self, student_list, teacher_list):
        self.num_levels = min(len(student_list), len(teacher_list))
        self.adapters = nn.ModuleList()
        for i in range(self.num_levels):
            sC = student_list[i].shape[1]
            tC = teacher_list[i].shape[1]
            self.adapters.append(_maybe_layer_1x1(sC, tC))
        self._initialized = True

    def forward(self, student_list):
        assert self._initialized
        out = []
        for i in range(self.num_levels):
            out.append(self.adapters[i](student_list[i]))
        return out








