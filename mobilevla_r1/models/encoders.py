"""Frozen modality encoders. No randomly initialized frozen substitutes."""
import importlib
import torch
from torch import nn
from mobilevla_r1.checkpoints import load_tensors
from transformers import AutoConfig, AutoImageProcessor, CLIPVisionModel, SiglipVisionModel


class RGBEncoder(nn.Module):
    def __init__(self, path, layer=-2, feature="auto"):
        super().__init__()
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        kind = cfg.model_type
        if kind not in {"siglip_vision_model", "clip_vision_model"}:
            raise ValueError(f"Unsupported RGB tower {kind}; provide cached RGB features")
        cls = SiglipVisionModel if kind == "siglip_vision_model" else CLIPVisionModel
        self.encoder = cls.from_pretrained(path, local_files_only=True, use_safetensors=True)
        self.processor = AutoImageProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        if feature not in {"auto", "patch", "cls_patch"}:
            raise ValueError("RGB feature must be auto, patch, or cls_patch")
        self.layer = layer
        self.drop_cls = (kind == "clip_vision_model") if feature == "auto" else feature == "patch"
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, frames):
        pixels = self.processor(images=frames, return_tensors="pt")["pixel_values"]
        param = next(self.encoder.parameters())
        features = self.encoder(pixels.to(param.device, param.dtype), output_hidden_states=True).hidden_states[self.layer]
        return features[:, 1:] if self.drop_cls else features


class DepthAnythingEncoder(nn.Module):
    """Official DAV2 DINOv2 features from supplied depth maps (replicated to RGB).

    The map-to-encoder normalization is an explicit implementation choice because
    the manuscript does not specify it. Input depth maps must be finite [H,W].
    Install the official depth_anything_v2 package and supply its checkpoint.
    """
    def __init__(self, checkpoint, variant="vitl", image_size=518):
        super().__init__()
        from depth_anything_v2.dpt import DepthAnythingV2
        specs = {"vits": (64, [48, 96, 192, 384]), "vitb": (128, [96, 192, 384, 768]),
                 "vitl": (256, [256, 512, 1024, 1024])}
        width, channels = specs[variant]
        model = DepthAnythingV2(encoder=variant, features=width, out_channels=channels)
        model.load_state_dict(load_tensors(checkpoint), strict=True)
        self.encoder = model.pretrained
        self.image_size = image_size
        if image_size <= 0 or image_size % 14:
            raise ValueError("DAV2 image_size must be divisible by 14")
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, maps):
        from torch.nn import functional as F
        tensors = []
        for depth in maps:
            x = torch.as_tensor(depth, dtype=torch.float32)
            if x.ndim != 2 or not torch.isfinite(x).all():
                raise ValueError("Depth map must be finite [H,W]")
            x = (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)
            x = F.interpolate(x[None, None], size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            tensors.append(x.expand(-1, 3, -1, -1))
        x = torch.cat(tensors)
        mean, std = x.new_tensor([0.485, 0.456, 0.406]), x.new_tensor([0.229, 0.224, 0.225])
        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        param = next(self.encoder.parameters())
        return self.encoder.get_intermediate_layers(x.to(param.device, param.dtype), n=1)[0]


class PointTransformerV3Encoder(nn.Module):
    """Adapter for official Pointcept PointTransformerV3; strict pretrained load.

    kwargs must match the checkpoint architecture. No automatic key dropping.
    Arrays store xyz followed by exactly in_channels feature columns.
    """
    def __init__(self, checkpoint, kwargs, grid_size=0.02, state_prefix="", factory="pointcept.models.point_transformer_v3.point_transformer_v3m1_base:PointTransformerV3"):
        super().__init__()
        if not isinstance(grid_size, (int, float)) or not 0 < grid_size < float("inf"):
            raise ValueError("grid_size must be finite and positive")
        module, name = factory.split(":")
        self.encoder = getattr(importlib.import_module(module), name)(**kwargs)
        state = load_tensors(checkpoint)
        state = state.get("state_dict", state)
        if state_prefix:
            state = {k[len(state_prefix):]: v for k, v in state.items() if k.startswith(state_prefix)}
        self.encoder.load_state_dict(state, strict=True)
        self.channels = kwargs.get("in_channels", 6)
        self.grid_size = grid_size
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, clouds):
        param = next(self.encoder.parameters())
        results = []
        for cloud in clouds:
            points = torch.as_tensor(cloud, dtype=torch.float32, device=param.device)
            if points.ndim != 2 or points.size(1) != 3 + self.channels or not points.size(0):
                raise ValueError(f"Point input must be nonempty [N,{3 + self.channels}] = xyz + checkpoint features")
            if not torch.isfinite(points).all():
                raise ValueError("Point input contains NaN/Inf")
            data = {"coord": points[:, :3].contiguous(), "feat": points[:, 3:].to(param.dtype).contiguous(),
                    "offset": torch.tensor([points.size(0)], device=param.device, dtype=torch.long), "grid_size": self.grid_size}
            result = self.encoder(data)
            results.append(result.feat)
        return results


def build_encoders(cfg):
    result = nn.ModuleDict()
    for modality, spec in cfg.items():
        if spec["type"] == "cached":
            continue
        options = {k: v for k, v in spec.items() if k != "type"}
        classes = {"rgb": RGBEncoder, "depth_anything_v2": DepthAnythingEncoder,
                   "point_transformer_v3": PointTransformerV3Encoder}
        result[modality] = classes[spec["type"]](**options)
    return result
