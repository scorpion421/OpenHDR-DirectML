"""Neural network architecture for real-time SDR-to-HDR video upconversion.

Implements:
1. Global Color Gamut Expansion (Rec.709 to BT.2020).
2. Local Tone Mapping (LTM) & Highlight Enhancement (HE).
3. Differentiable SMPTE ST 2084 (PQ) transfer function for HDR10 output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# Standard Rec.709 to BT.2020 linear color transformation matrix
REC709_TO_BT2020_MATRIX = [
    [0.6274040786, 0.3292820974, 0.0433137976],
    [0.0690972332, 0.9195403953, 0.0113611893],
    [0.0163914389, 0.0880133075, 0.8955952536],
]

# SMPTE ST 2084 (PQ) constants
PQ_M1 = 2610.0 / 16384.0          # 0.1593017578125
PQ_M2 = (2523.0 / 4096.0) * 128.0 # 78.84375
PQ_C1 = 3424.0 / 4096.0           # 0.8359375
PQ_C2 = (2413.0 / 4096.0) * 32.0  # 18.8515625
PQ_C3 = (2392.0 / 4096.0) * 32.0  # 18.6875


def linearize_rec709(x: torch.Tensor) -> torch.Tensor:
    """Converts gamma-encoded Rec.709/sRGB [0, 1] to linear luminance [0, 1]."""
    # IEC 61966-2-1 / standard approx gamma 2.4 with linear toe
    threshold = 0.04045
    linear = torch.where(
        x <= threshold,
        x / 12.92,
        torch.pow(torch.clamp((x + 0.055) / 1.055, min=1e-6), 2.4),
    )
    return linear


def linear_rec709_to_bt2020(x: torch.Tensor) -> torch.Tensor:
    """Applies Rec.709 to BT.2020 linear transformation matrix.
    
    x: Tensor of shape (B, 3, H, W)
    """
    matrix = x.new_tensor(REC709_TO_BT2020_MATRIX).view(1, 3, 3, 1, 1)
    # Broadcast multiply across channels
    r = x[:, 0:1] * matrix[:, 0, 0:1] + x[:, 1:2] * matrix[:, 0, 1:2] + x[:, 2:3] * matrix[:, 0, 2:3]
    g = x[:, 0:1] * matrix[:, 1, 0:1] + x[:, 1:2] * matrix[:, 1, 1:2] + x[:, 2:3] * matrix[:, 1, 2:3]
    b = x[:, 0:1] * matrix[:, 2, 0:1] + x[:, 1:2] * matrix[:, 2, 1:2] + x[:, 2:3] * matrix[:, 2, 2:3]
    return torch.cat([r, g, b], dim=1)


def linear_to_pq(x: torch.Tensor, target_peak_nits: float = 1000.0) -> torch.Tensor:
    """Applies SMPTE ST 2084 PQ curve.
    
    Input x is normalized linear luminance [0, 1] representing [0, target_peak_nits].
    Output is PQ non-linear code values in [0, 1].
    Reference max luminance for PQ is 10,000 nits.
    """
    # Scale linear light relative to 10,000 nits
    scale = target_peak_nits / 10000.0
    lum = torch.clamp(x * scale, min=1e-8, max=1.0)
    
    y = torch.pow(lum, PQ_M1)
    num = PQ_C1 + PQ_C2 * y
    den = 1.0 + PQ_C3 * y
    pq = torch.pow(num / den, PQ_M2)
    return pq


class ConvBlock(nn.Module):
    """Depthwise separable convolution block for low-latency inference."""
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)))


class ResBlock(nn.Module):
    """Lightweight residual block."""
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


class RealTimeOpenHDRNet(nn.Module):
    """Real-time neural network for SDR-to-HDR10 conversion.
    
    Combines:
    - Global Color Gamut Mapping (GCM)
    - Dynamic Local Tone Mapping (LTM)
    - Specular Highlight Enhancement (HE)
    - Hardware-optimized FP16 DirectML execution graph
    """
    def __init__(
        self,
        base_channels: int = 16,
        num_resblocks: int = 3,
        target_peak_nits: float = 1000.0,
    ) -> None:
        super().__init__()
        self.target_peak_nits = target_peak_nits

        # 1. Feature Ingestion (RGB Rec.709)
        self.in_conv = nn.Conv2d(3, base_channels, kernel_size=3, padding=1, bias=True)
        self.in_act = nn.LeakyReLU(0.2, inplace=True)

        # 2. Residual feature extraction for local highlight and tone adjustment
        res_blocks = [ResBlock(base_channels) for _ in range(num_resblocks)]
        self.body = nn.Sequential(*res_blocks)

        # 3. Tone & Highlight Reconstruction Heads
        # Head A: Multiplicative gain map (LTM)
        self.gain_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
            nn.Sigmoid(),
        )

        # Head B: Additive specular highlight refinement (HE)
        self.highlight_head = nn.Sequential(
            nn.Conv2d(base_channels, base_channels // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels // 2, 3, kernel_size=1),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        # Initialize gain map close to 1.0 and highlight close to 0.0 for stable initial output
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode="fan_in")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Bias the final gain layer so sigmoid starts near natural expansion
        if hasattr(self.gain_head[-2], "bias") and self.gain_head[-2].bias is not None:
            nn.init.constant_(self.gain_head[-2].bias, 1.0)
        # Bias the highlight layer small
        if hasattr(self.highlight_head[-2], "bias") and self.highlight_head[-2].bias is not None:
            nn.init.constant_(self.highlight_head[-2].bias, 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input SDR tensor (B, 3, H, W) in range [0.0, 1.0], Rec.709 gamma.
        Returns:
            HDR10 PQ code values in range [0.0, 1.0], BT.2020 color primaries.
        """
        # Linearize input SDR luminance
        x_lin = linearize_rec709(x)

        # Extract features for dynamic tone mapping
        feat = self.in_act(self.in_conv(x))
        feat = self.body(feat)

        # Multiplicative tone expansion gain (1.0 to 3.0x dynamic range stretch)
        gain = 1.0 + 2.0 * self.gain_head(feat)

        # Additive specular highlight boost (for sun, lamps, reflections)
        highlight = 0.5 * self.highlight_head(feat)

        # Apply expansion in linear space
        lin_expanded = x_lin * gain + highlight

        # Gamut mapping: Convert from Rec.709 to BT.2020 color primaries
        bt2020_lin = linear_rec709_to_bt2020(lin_expanded)
        bt2020_lin = torch.clamp(bt2020_lin, min=0.0, max=1.0)

        # Apply SMPTE ST 2084 PQ curve
        pq_out = linear_to_pq(bt2020_lin, target_peak_nits=self.target_peak_nits)
        return pq_out
