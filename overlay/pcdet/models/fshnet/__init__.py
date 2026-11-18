"""FSHNet (Liu et al., "FSHNet: Fully Sparse Hybrid Network for 3D Object Detection", CVPR 2025),
ported to this repo's OpenPCDet version for Diver3D. Source: https://github.com/Say2L/FSHNet
(commit 8e8f23e, Apache-2.0, see LICENSE_FSHNET). Changes are listed in each file's header and in
docs/HYBRID_PLAN.md. Modules are imported by the registries (backbones_3d, dense_heads, detectors),
not here, so importing this package costs nothing.
"""
