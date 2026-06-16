def __getattr__(name):
    if name == "DiffusionModel":
        from .diffusion import DiffusionModel
        return DiffusionModel
    if name == "NoiseSchedule":
        from .noise_schedule import NoiseSchedule
        return NoiseSchedule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DiffusionModel", "NoiseSchedule"]
