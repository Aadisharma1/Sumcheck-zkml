import sys, struct, copy, pathlib
import torch, torch.nn as nn, torchvision, torchvision.transforms as T
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = pathlib.Path("fused_weights")
EPS = 1e-5
ATOL = 1e-4
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)


def _fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    gamma  = bn.weight.data
    beta   = bn.bias.data
    mu     = bn.running_mean
    var    = bn.running_var
    denom  = torch.sqrt(var + bn.eps)
    scale  = gamma / denom

    fused = nn.Conv2d(
        conv.in_channels, conv.out_channels, conv.kernel_size,
        stride=conv.stride, padding=conv.padding, bias=True,
        groups=conv.groups, dilation=conv.dilation,
    ).to(conv.weight.device)

    fused.weight.data = conv.weight.data * scale.reshape(-1, 1, 1, 1)
    if conv.bias is not None:
        fused.bias.data = (conv.bias.data - mu) * scale + beta
    else:
        fused.bias.data = -mu * scale + beta
    return fused


def _fuse_sequential(seq: nn.Sequential) -> nn.Sequential:
    layers = list(seq.children())
    fused_layers = []
    i = 0
    while i < len(layers):
        if isinstance(layers[i], nn.Conv2d) and i + 1 < len(layers) and isinstance(layers[i + 1], nn.BatchNorm2d):
            fused_layers.append(_fuse_conv_bn(layers[i], layers[i + 1]))
            i += 2
        else:
            fused_layers.append(layers[i])
            i += 1
    return nn.Sequential(*fused_layers)


def _fuse_basicblock(block):
    if hasattr(block, 'conv1') and hasattr(block, 'bn1'):
        block.conv1 = _fuse_conv_bn(block.conv1, block.bn1)
        block.bn1 = nn.Identity()
    if hasattr(block, 'conv2') and hasattr(block, 'bn2'):
        block.conv2 = _fuse_conv_bn(block.conv2, block.bn2)
        block.bn2 = nn.Identity()
    if block.downsample is not None:
        block.downsample = _fuse_sequential(block.downsample)
    return block


def fuse_resnet18(model: nn.Module) -> nn.Module:
    model = copy.deepcopy(model)
    model.eval()

    if hasattr(model, 'conv1') and hasattr(model, 'bn1'):
        model.conv1 = _fuse_conv_bn(model.conv1, model.bn1)
        model.bn1 = nn.Identity()

    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for i in range(len(layer)):
            layer[i] = _fuse_basicblock(layer[i])

    return model


def _dump_tensor(t: torch.Tensor, path: pathlib.Path):
    flat = t.detach().cpu().float().numpy().flatten()
    with open(path, 'wb') as f:
        f.write(struct.pack(f'<{len(flat)}f', *flat))


def export_weights(model: nn.Module, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = 0
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            tag = f"{idx:03d}_{name.replace('.', '_')}"
            _dump_tensor(mod.weight, out_dir / f"{tag}_weight.bin")
            if mod.bias is not None:
                _dump_tensor(mod.bias, out_dir / f"{tag}_bias.bin")

            meta = {
                'c_out': mod.out_channels, 'c_in': mod.in_channels,
                'kh': mod.kernel_size[0], 'kw': mod.kernel_size[1],
                'sh': mod.stride[0], 'sw': mod.stride[1],
                'ph': mod.padding[0], 'pw': mod.padding[1],
                'groups': mod.groups,
            }
            with open(out_dir / f"{tag}_meta.txt", 'w') as f:
                for k, v in meta.items():
                    f.write(f"{k}={v}\n")
            idx += 1
        elif isinstance(mod, nn.Linear):
            tag = f"{idx:03d}_{name.replace('.', '_')}"
            _dump_tensor(mod.weight, out_dir / f"{tag}_weight.bin")
            if mod.bias is not None:
                _dump_tensor(mod.bias, out_dir / f"{tag}_bias.bin")
            with open(out_dir / f"{tag}_meta.txt", 'w') as f:
                f.write(f"out_features={mod.out_features}\n")
                f.write(f"in_features={mod.in_features}\n")
            idx += 1


def export_input(img: torch.Tensor, out_dir: pathlib.Path):
    _dump_tensor(img, out_dir / "input.bin")
    with open(out_dir / "input_meta.txt", 'w') as f:
        f.write(f"shape={'x'.join(str(d) for d in img.shape)}\n")


def verify_equivalence(orig: nn.Module, fused: nn.Module, x: torch.Tensor) -> bool:
    orig.eval()
    fused.eval()
    with torch.no_grad():
        y_orig  = orig(x)
        y_fused = fused(x)
    diff = (y_orig - y_fused).abs().max().item()
    ok = diff < ATOL
    sys.stderr.write(f"max |y_orig - y_fused| = {diff:.2e}  {'PASS' if ok else 'FAIL'}\n")
    return ok


def build_cifar_resnet18(pretrained=True):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model.to(DEVICE)


def main():
    model = build_cifar_resnet18(pretrained=True)
    model.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize(CIFAR_MEAN, CIFAR_STD)])
    ds = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    img, label = ds[0]
    x = img.unsqueeze(0).to(DEVICE)

    fused = fuse_resnet18(model)
    fused.to(DEVICE)

    if not verify_equivalence(model, fused, x):
        sys.stderr.write("FATAL: fusion verification failed\n")
        sys.exit(1)

    export_weights(fused, OUT_DIR)
    export_input(x, OUT_DIR)

    with torch.no_grad():
        logits = fused(x)
    pred = logits.argmax(dim=1).item()
    _dump_tensor(logits, OUT_DIR / "expected_output.bin")

    sys.stderr.write(f"predicted_class={pred} true_label={label}\n")
    sys.stderr.write(f"exported {len(list(OUT_DIR.glob('*.bin')))} bin files to {OUT_DIR}\n")


if __name__ == '__main__':
    main()
