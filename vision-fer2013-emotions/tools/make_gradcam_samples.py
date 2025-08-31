import os, torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
from src.models import make_resnet18
from src.data import make_test_loader

def gradcam(model, x, target_layer):
    """Generate Grad-CAM for a single image tensor x (1,C,H,W)."""
    feats, grads = [], []

    def fwd_hook(_, __, out): feats.append(out)
    def bwd_hook(_, gin, gout): grads.append(gout[0])
    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    logits = model(x)
    cls = logits.argmax(1).item()
    logits[0,cls].backward()

    g = grads[0].mean(dim=(2,3), keepdim=True)
    w = (feats[0] * g).sum(dim=1, keepdim=True)
    w = torch.relu(w)
    w = torch.nn.functional.interpolate(w, size=x.shape[2:], mode="bilinear", align_corners=False)
    w = (w - w.min()) / (w.max() - w.min() + 1e-8)

    handle_fwd.remove(); handle_bwd.remove()
    return cls, w[0,0].detach().cpu().numpy()


def overlay(img, mask):
    """Overlay heatmap on image (PIL grayscale to RGB)."""
    img = np.array(img.convert("RGB")).astype(float)/255.0
    cmap = plt.get_cmap("jet")(mask)[..., :3]
    overlayed = (0.6*img + 0.4*cmap)
    overlayed = np.clip(overlayed,0,1)
    return (overlayed*255).astype(np.uint8)

if __name__=="__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="data/fer")
    ap.add_argument("--out-dir", type=str, default="artifacts/gradcam")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader, class_to_idx = make_test_loader(root=args.data_root, img_size=48, batch_size=1)
    labels = [c for c,_ in sorted(class_to_idx.items(), key=lambda kv: kv[1])]
    model = make_resnet18(len(labels), grayscale=True).to(device)
    model.load_state_dict(torch.load("models/resnet18.pt", map_location=device))
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    conv_layer = model.layer4[1].conv2

    for i,(x,y) in enumerate(test_loader):
        if i>=10: break  # limit to 10 samples
        x = x.to(device)
        cls, mask = gradcam(model, x, conv_layer)
        img = transforms.ToPILImage()(x[0].cpu())
        out = overlay(img, mask)
        out_path = os.path.join(args.out_dir, f"sample_{i}_true-{labels[y.item()]}_pred-{labels[cls]}.png")
        Image.fromarray(out).save(out_path)
        print("Saved", out_path)
