from pathlib import Path
from loguru import logger


LOG_DIR = Path("logs")

LOG_DIR.mkdir(
    exist_ok=True
)


logger.remove()


logger.add(
    LOG_DIR / "app.log",
    rotation="5 MB",
    retention="10 days",
    level="INFO",
    format="{time} | {level} | {message}"
)


logger.add(
    lambda message: print(message, end=""),
    level="INFO"
)