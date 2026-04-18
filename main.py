import os
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from dotenv import load_dotenv

# Load .env BEFORE importing engine modules (they may read env at import time)
load_dotenv()

from engine.data_fetcher import fetch_market_data
from engine.forecaster import ChronosEngine
from engine.tele_reporter import send_alert_to_telegram

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_SYMBOL = os.getenv("TARGET_SYMBOL", "GC=F")
MODEL_ID = os.getenv("MODEL_ID", "amazon/chronos-2")
UPDATE_INTERVAL_SEC = int(os.getenv("UPDATE_INTERVAL_SEC", str(4 * 3600)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("chronos-forecaster")

# ---------------------------------------------------------------------------
# Global model reference (loaded once during lifespan)
# ---------------------------------------------------------------------------
model_engine: ChronosEngine | None = None


# ---------------------------------------------------------------------------
# Lifespan (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_engine
    log.info("Starting up — loading Chronos model...")
    model_engine = ChronosEngine(model_id=MODEL_ID)
    log.info("Model ready. Starting background scheduler.")

    # Launch periodic task
    task = asyncio.create_task(_periodic_forecast())
    yield

    # Shutdown
    task.cancel()
    log.info("Shutting down.")


app = FastAPI(
    title="Chronos Forecaster",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Background scheduler
# ---------------------------------------------------------------------------
async def _periodic_forecast():
    """Run forecast + Telegram push every UPDATE_INTERVAL_SEC."""

    # Small initial delay so the first run doesn't race with startup
    await asyncio.sleep(5)

    while True:
        try:
            log.info(f"Scheduled run: {TARGET_SYMBOL}")

            # Offload blocking I/O + CPU work to a thread so FastAPI stays responsive
            context_df = await asyncio.to_thread(
                fetch_market_data, symbol=TARGET_SYMBOL, interval="1h", period="15d"
            )
            analysis, chart_bytes = await asyncio.to_thread(
                model_engine.forecast_and_plot, context_df, 24
            )
            await asyncio.to_thread(
                send_alert_to_telegram, analysis, chart_bytes
            )

            log.info("Scheduled run complete.")

        except Exception:
            log.exception("Error in scheduled forecast")

        await asyncio.sleep(UPDATE_INTERVAL_SEC)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_ID,
        "target": TARGET_SYMBOL,
        "interval_sec": UPDATE_INTERVAL_SEC,
    }


@app.get("/forecast")
def forecast(symbol: str = None, prediction_length: int = 24):
    """
    On-demand forecast. Returns JSON with analysis text.
    AG-Mini Trading Bot calls this endpoint for confirmation signals.
    """
    target = symbol or TARGET_SYMBOL
    try:
        context_df = fetch_market_data(symbol=target, interval="1h", period="15d")
        analysis, _ = model_engine.forecast_and_plot(context_df, prediction_length=prediction_length)
        return {"status": "success", "symbol": target, "analysis": analysis}
    except Exception as e:
        log.exception(f"Forecast failed for {target}")
        return {"status": "error", "message": str(e)}


@app.get("/chart")
def chart(symbol: str = None):
    """Returns the forecast chart as a PNG image."""
    target = symbol or TARGET_SYMBOL
    try:
        context_df = fetch_market_data(symbol=target, interval="1h", period="15d")
        _, png_bytes = model_engine.forecast_and_plot(context_df, prediction_length=24)
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        log.exception(f"Chart generation failed for {target}")
        return Response(content=str(e), status_code=500, media_type="text/plain")
