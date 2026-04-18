import yfinance as yf
import pandas as pd
import time


def fetch_market_data(
    symbol: str = "GC=F",
    interval: str = "1h",
    period: str = "15d",
    max_retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV from Yahoo Finance.
    Returns DataFrame with columns: timestamp, target (Close price).
    Includes retry logic for transient network / rate-limit errors.
    """
    last_err = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"[DataFetcher] Fetching {symbol} ({interval}, {period}) — attempt {attempt}")
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for {symbol}")

            df = df.reset_index()

            # yfinance names the time column "Datetime" for intraday, "Date" for daily.
            time_col = None
            for candidate in ("Datetime", "Date", "datetime", "date"):
                if candidate in df.columns:
                    time_col = candidate
                    break

            if time_col is None:
                # Fallback: use the first column if it looks like a timestamp
                first_col = df.columns[0]
                if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                    time_col = first_col
                else:
                    raise ValueError(
                        f"Cannot find timestamp column. Columns: {list(df.columns)}"
                    )

            result = pd.DataFrame({
                "timestamp": df[time_col],
                "target": df["Close"],
            })

            # Drop NaN rows (weekends, gaps)
            result = result.dropna(subset=["target"])
            result = result.sort_values("timestamp").reset_index(drop=True)

            print(f"[DataFetcher] Got {len(result)} data points for {symbol}")
            return result

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                wait = 2 ** attempt  # exponential backoff: 2s, 4s
                print(f"[DataFetcher] Attempt {attempt} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    raise RuntimeError(f"Failed to fetch {symbol} after {max_retries} attempts: {last_err}")


if __name__ == "__main__":
    df = fetch_market_data()
    print(f"Rows: {len(df)}")
    print(df.tail(5))
