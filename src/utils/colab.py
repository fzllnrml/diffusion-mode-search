from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def mount_drive_if_needed(
    local_dir: str = "./checkpoints",
    drive_subdir: str = "MyDrive/thesis/checkpoints",
    mount_point: str = "/content/drive",
) -> str:
    mount_path = Path(mount_point)

    if mount_path.exists() and (mount_path / "MyDrive").exists():
        drive_dir = mount_path / drive_subdir
        drive_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Google Drive уже смонтирован: %s", drive_dir)
        return str(drive_dir)

    try:
        from google.colab import drive
        drive.mount(str(mount_point), force_remount=False)
        drive_dir = mount_path / drive_subdir
        drive_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Google Drive смонтирован: %s", drive_dir)
        return str(drive_dir)

    except ImportError:
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        logger.info("Не Colab: чекпоинты в %s", local)
        return str(local)

    except Exception as e:
        logger.warning("Не удалось смонтировать Drive (%s). Используем %s", e, local_dir)
        local = Path(local_dir)
        local.mkdir(parents=True, exist_ok=True)
        return str(local)


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False
