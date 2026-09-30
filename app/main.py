from __future__ import annotations

import logging

from fastapi import FastAPI

from app.api import router
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = FastAPI(title=settings.app_name, version="0.1.0")
app.include_router(router)
