from .clustering import merge_close, agglomerative_merge


def __getattr__(name):
    if name == "ModeFinder":
        from .mode_finder import ModeFinder
        return ModeFinder
    if name == "ModeFinderV2":
        from .mode_finder_v2 import ModeFinderV2
        return ModeFinderV2
    if name == "CoarseToFineFinder":
        from .mode_finder_v2 import CoarseToFineFinder
        return CoarseToFineFinder
    if name == "BaselineModeFinder":
        from .baseline import BaselineModeFinder
        return BaselineModeFinder
    if name == "ModeFinderV3F1":
        from .mode_finder_v3_f1 import ModeFinderV3F1
        return ModeFinderV3F1
    if name == "ModeFinderV3F2":
        from .mode_finder_v3_f2 import ModeFinderV3F2
        return ModeFinderV3F2
    if name == "ModeFinderV3F3":
        from .mode_finder_v3_f3 import ModeFinderV3F3
        return ModeFinderV3F3
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ModeFinder", "ModeFinderV2", "CoarseToFineFinder",
    "BaselineModeFinder", "merge_close", "agglomerative_merge",
    "ModeFinderV3F1", "ModeFinderV3F2", "ModeFinderV3F3",
]
