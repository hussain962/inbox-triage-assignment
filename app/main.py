from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.services.triage import build_triage_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.triage = build_triage_service()
    yield


def create_app() -> FastAPI:
    app = FastAPI(title="Inbox triage", lifespan=lifespan)
    app.include_router(router)
    return app


app = create_app()
