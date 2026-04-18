import matplotlib
matplotlib.use('Agg')  # Headless server — no GUI
import matplotlib.pyplot as plt
import torch
import numpy as np
import io
import pandas as pd
from typing import Tuple


class ChronosEngine:
    """Wrapper around Amazon Chronos-2 for time-series forecasting."""

    def __init__(self, model_id: str = "amazon/chronos-2"):
        print(f"[Chronos] Loading model: {model_id} on CPU...")

        # Chronos-2 uses Chronos2Pipeline; Bolt/T5 use BaseChronosPipeline.
        # We auto-detect to stay forwards-compatible.
        try:
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                model_id,
                device_map="cpu",
                dtype=torch.float32,
            )
            self._use_df_api = True          # predict_df()
            print("[Chronos] Loaded via Chronos2Pipeline (predict_df API)")
        except (ImportError, Exception):
            from chronos import BaseChronosPipeline
            self.pipeline = BaseChronosPipeline.from_pretrained(
                model_id,
                device_map="cpu",
                dtype=torch.float32,
            )
            self._use_df_api = False         # predict() tensor API
            print("[Chronos] Loaded via BaseChronosPipeline (tensor API)")

        self.model_id = model_id
        print("[Chronos] Model loaded successfully!")

    # ------------------------------------------------------------------
    def forecast_and_plot(
        self,
        context_df: pd.DataFrame,
        prediction_length: int = 24,
    ) -> Tuple[str, bytes]:
        """
        Run forecast on a (timestamp, target) dataframe.
        Returns (analysis_text, png_bytes).
        """
        values = context_df["target"].dropna().values.astype(np.float32)

        if len(values) < 10:
            raise ValueError(f"Not enough data points ({len(values)}). Need >= 10.")

        print(f"[Chronos] Generating forecast for {prediction_length} steps "
              f"from {len(values)} context points...")

        if self._use_df_api:
            # --- Chronos-2 DataFrame API ---
            # Chronos-2 requires a uniform, gapless DatetimeIndex
            ts = context_df.set_index("timestamp")["target"].dropna()
            ts.index = pd.to_datetime(ts.index)
            ts = ts.sort_index()

            # Detect dominant freq (most common interval), then resample to fill gaps
            diffs = ts.index.to_series().diff().dropna()
            dominant = diffs.mode()[0]
            minutes = int(dominant.total_seconds() // 60)
            freq_alias = {60: "h", 240: "4h", 1440: "D"}.get(minutes, f"{minutes}min")

            # Resample & forward-fill gaps — this also sets index.freq
            ts = ts.resample(freq_alias).last().ffill()

            ctx = ts.reset_index()
            ctx.columns = ["timestamp", "target"]
            ctx["id"] = "series_0"

            # NOTE: do NOT pass freq= kwarg — Chronos-2 infers it from the index
            pred_df = self.pipeline.predict_df(
                ctx,
                prediction_length=prediction_length,
                quantile_levels=[0.1, 0.5, 0.9],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )

            q10 = pred_df["0.1"].values
            q50 = pred_df["0.5"].values
            q90 = pred_df["0.9"].values

        else:
            # --- Bolt / T5 tensor API ---
            context_tensor = torch.tensor(values).unsqueeze(0)   # [1, T]
            forecast = self.pipeline.predict(
                context_tensor,
                prediction_length=prediction_length,
            )
            # forecast is a Tensor [1, num_samples, prediction_length]
            if isinstance(forecast, torch.Tensor):
                forecast_np = forecast.numpy()[0]
            else:
                # some builds return a list of tensors
                forecast_np = np.stack([f.numpy() for f in forecast])[0]

            q10, q50, q90 = np.quantile(forecast_np, [0.1, 0.5, 0.9], axis=0)

        # --- Build chart ---
        show = min(len(values), 120)
        hist_vals = values[-show:]
        hist_ts = list(range(show))
        fut_ts = list(range(show, show + prediction_length))

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')

        ax.plot(hist_ts, hist_vals, color='#00d2ff', linewidth=1.4, label='Historical')
        ax.plot(fut_ts, q50, color='#ff6ec7', linewidth=2, label='Forecast (Median)')
        ax.fill_between(fut_ts, q10, q90, alpha=0.25, color='#ff6ec7', label='P10-P90')

        ax.set_title(f'Chronos AI Forecast  |  {self.model_id}',
                     color='white', fontsize=13, fontweight='bold', pad=12)
        ax.set_xlabel('Time Steps (H1 candles)', color='#aaa', fontsize=10)
        ax.set_ylabel('Price (USD)', color='#aaa', fontsize=10)
        ax.tick_params(colors='#888')
        ax.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white', fontsize=9)
        ax.grid(True, alpha=0.15, color='#555')
        for spine in ax.spines.values():
            spine.set_color('#333')

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        png_bytes = buf.read()
        plt.close(fig)

        # --- Text analysis ---
        current = float(values[-1])
        median_end = float(q50[-1])
        pct = ((median_end - current) / current) * 100
        trend = "TANG" if pct > 0.05 else ("GIAM" if pct < -0.05 else "DI NGANG")

        analysis = (
            f"[Chronos AI Forecast]\n"
            f"Gia hien tai: {current:.2f}\n"
            f"Du doan {prediction_length} nen toi (Median): {median_end:.2f}\n"
            f"Xu Huong: {trend} ({pct:+.2f}%)\n"
            f"Vung rui ro (P10-P90): [{float(q10[-1]):.2f} - {float(q90[-1]):.2f}]"
        )

        return analysis, png_bytes
