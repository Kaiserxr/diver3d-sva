"""Registry entries for FSHNet that import its code only when an FSHNet model is built.

OpenPCDet imports everything listed in its registries whenever any model starts. Registering these
small factories instead of the real classes means a problem in the FSHNet port can only affect
FSHNet runs, never the other models (same idea as PTv3Backbone's lazy import).
"""


def FSHNetDiver3D(*args, **kwargs):
    from .backbone import FSHNetDiver3D as cls
    return cls(*args, **kwargs)


def SparseDynamicHead(*args, **kwargs):
    from .head import SparseDynamicHead as cls
    return cls(*args, **kwargs)


def FSHNetCenterPoint(*args, **kwargs):
    from .detector import FSHNetCenterPoint as cls
    return cls(*args, **kwargs)
