from torch.nn.utils import prune as nnprune

def should_skip_module(name: str) -> bool:
    if ("rpn.head.cls_logits" in name) or ("rpn.head.bbox_pred" in name) or ("roi_heads.box_predictor" in name):
        return True
    if (".block.2.fc1" in name) or (".block.2.fc2" in name):  # SE MLPs
        return True
    if name.startswith("backbone.fpn"):
        return True
    return False

def prune_convs_structured(mods, amount):
    for m, name in mods:
        try: nnprune.ln_structured(m, name="weight", amount=amount, n=1, dim=0)
        except Exception as e: print(f"[WARN] conv prune skip {name}: {e}")

def prune_linears_unstructured(mods, amount):
    for m, name in mods:
        try: nnprune.l1_unstructured(m, name="weight", amount=amount)
        except Exception as e: print(f"[WARN] linear prune skip {name}: {e}")

def tensor_sparsity(t: torch.Tensor) -> float:
    nz = (t != 0).sum().item()
    total = t.numel()
    return 1.0 - (nz / max(1, total))

def report_sparsity(m: nn.Module):
    all_s, n = 0.0, 0
    conv_s, n_conv = 0.0, 0
    lin_s,  n_lin  = 0.0, 0
    for name, mod in m.named_modules():
        if hasattr(mod, "weight") and isinstance(mod.weight, torch.Tensor):
            s = tensor_sparsity(mod.weight.data)
            all_s += s; n += 1
            if isinstance(mod, nn.Conv2d):
                conv_s += s; n_conv += 1
            elif isinstance(mod, nn.Linear):
                lin_s += s; n_lin += 1
    print(f"[REPORT] avg sparsity all={all_s/max(1,n):.3f} | conv={conv_s/max(1,n_conv):.3f} | linear={lin_s/max(1,n_lin):.3f}")


