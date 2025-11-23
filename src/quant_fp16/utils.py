import torchvision.transforms as T
from PIL import Image
import glob
to_tensor = T.Compose([T.ToTensor()])  # PIL -> FloatTensor CxHxW in [0,1]

def load_calib_images(max_imgs=256):
    paths = sorted(glob.glob(str(Path(calib_images_dir) / "*")))[:max_imgs]
    imgs = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            imgs.append(to_tensor(img))
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return imgs

# Keep normalization layers in float32 for numerical stability
def norms_to_fp32(m: nn.Module):
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        m.float()
    for c in m.children():
        norms_to_fp32(c)


# Also cast Linear layers to half (weight+bias), since we’re on GPU
def linears_to_half(m: nn.Module):
    for n, c in m.named_children():
        linears_to_half(c)
    if isinstance(m, nn.Linear):
        m.half()


def run_quant_infer(m, imgs, warmup=3, iters=10):
    m.eval()
    use_amp = (next(m.parameters()).device.type == "cuda")

    with torch.inference_mode():
        # warmup
        for _ in range(warmup):
            img = imgs[0].to(next(m.parameters()).device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _ = m([img])
        # time
        t0 = time.time()
        for i in range(min(iters, len(imgs))):
            img = imgs[i].to(next(m.parameters()).device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                _ = m([img])
        t1 = time.time()
    return (t1 - t0) / min(iters, len(imgs))