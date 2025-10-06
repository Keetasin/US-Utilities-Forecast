# src/spark/applications/spark_jobs/pull_yf.py

import argparse
import os
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from pyspark.sql import SparkSession

# -----------------------------
# Config
# -----------------------------
OUTPUT_DIR = "/opt/airflow/data/spark_out"
OUT_PATH = os.path.join(OUTPUT_DIR, "market.parquet")
EXOG_OUT_PATH = os.path.join(OUTPUT_DIR, "exog.parquet")


# -----------------------------
# Helpers
# -----------------------------
def _flatten_single_ticker_cols(df: pd.DataFrame) -> pd.DataFrame:
    """ทำให้คอลัมน์เป็น single-level เสมอ + normalize ชื่อยอดฮิต"""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in tup if x not in ("", None)])
            for tup in df.columns
        ]
    rename_map = {
        "Adj Close": "Adj Close",
        "adj close": "Adj Close",
        "adj_close": "Adj Close",
        "close": "Close",
    }
    df.columns = [rename_map.get(c, c) for c in df.columns]
    return df


def _pick_close_column(cols) -> str | None:
    """เลือกคอลัมน์ราคาปิด เช่น Close, Close_AEP, AEP_Close"""
    close_like = [c for c in cols if "close" in c.lower()]
    if not close_like:
        return None
    for c in close_like:
        if "adj" not in c.lower():
            return c
    return close_like[0]


def dl_close(ticker: str, period: str) -> pd.DataFrame:
    """ดาวน์โหลดราคาปิดของหุ้น"""
    df = yf.download(
        ticker,
        period=period,
        group_by="column",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"[yfinance] empty dataframe for {ticker} period={period}")

    df = _flatten_single_ticker_cols(df)
    close_col = _pick_close_column(df.columns)
    if not close_col:
        raise KeyError(f"No close-like column in df.columns={list(df.columns)} for {ticker}")

    df = df[[close_col]].reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df = df.rename(columns={close_col: "close"})

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .set_index("date")
        .asfreq("B")
        .ffill()
        .bfill()
        .reset_index()
    )
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0)
    return df[["date", "close"]]


def dl_exog(code: str, period: str, out_col: str) -> pd.DataFrame:
    """ดาวน์โหลดตัวแปรภายนอก (exogenous variables)"""
    df = yf.download(
        code,
        period=period,
        group_by="column",
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if df is None or len(df) == 0:
        raise RuntimeError(f"[yfinance exog] empty dataframe for {code}")

    df = _flatten_single_ticker_cols(df)
    close_col = _pick_close_column(df.columns)
    if not close_col:
        raise KeyError(f"No close-like column in exog df.columns={list(df.columns)} for {code}")

    df = df[[close_col]].reset_index()
    if "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    df = df.rename(columns={close_col: out_col})

    df["date"] = pd.to_datetime(df["date"])
    df = (
        df.sort_values("date")
        .set_index("date")
        .asfreq("B")
        .ffill()
        .bfill()
        .reset_index()
    )
    df[out_col] = pd.to_numeric(df[out_col], errors="coerce").fillna(0)
    return df[["date", out_col]]


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_spark(app_name: str = "pull-yfinance-to-parquet") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Download stock prices + exog and save as Parquet")
    parser.add_argument("--period", type=str, default="5y", help="yfinance period, e.g. 6mo, 2y, 5y")
    parser.add_argument("--tickers", type=str, default="AEP", help="comma/space separated tickers")
    parser.add_argument("--exog", type=str, default="CL=F,NG=F,XLU", help="comma separated exogenous codes")
    args, _ = parser.parse_known_args()

    tickers = [t.strip() for t in args.tickers.replace(",", " ").split() if t.strip()]
    exog_list = [e.strip() for e in args.exog.split(",") if e.strip()]

    print(f"[Args] period={args.period} tickers={tickers} exog={exog_list}")

    # ---------- Download prices ----------
    price_frames = []
    for t in tickers:
        df = dl_close(t, args.period)
        df["symbol"] = t
        price_frames.append(df)
    if not price_frames:
        raise RuntimeError("No price data collected")

    price_long = pd.concat(price_frames, ignore_index=True)
    price_long = (
        price_long.sort_values(["date", "symbol"])
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["close"])
    )

    # ---------- Download exog ----------
    exog_wide = None
    if exog_list:
        exogs = []

        def safe_colname(code: str) -> str:
            return (
                code.replace("=", "_")
                .replace("^", "IDX_")
                .replace("-", "_")
                .replace("/", "_")
                .replace(" ", "_")
            ).lower()

        for code in exog_list:
            col = safe_colname(code)
            df_e = dl_exog(code, args.period, out_col=col)
            exogs.append(df_e)

        if exogs:
            exog_wide = exogs[0]
            for i in range(1, len(exogs)):
                exog_wide = exog_wide.merge(exogs[i], on="date", how="outer")

            exog_wide = (
                exog_wide.sort_values("date")
                .set_index("date")
                .asfreq("B")
                .ffill()
                .bfill()
                .reset_index()
            )
            exog_wide = exog_wide.fillna(0).replace([np.inf, -np.inf], 0)

    # รวม exog เข้ากับราคา (ซ้ายยาว)
    df_out = price_long.merge(exog_wide, on="date", how="left") if exog_wide is not None else price_long
    df_out = df_out.fillna(0).replace([np.inf, -np.inf], 0)

    # ---------- Write to Parquet (Spark) ----------
    ensure_output_dir()
    spark = build_spark()

    df_out["date"] = pd.to_datetime(df_out["date"])
    df_out["symbol"] = df_out["symbol"].astype(str)
    df_out["close"] = pd.to_numeric(df_out["close"], errors="coerce").fillna(0)

    base_cols = ["date", "symbol", "close"]
    exog_cols = [c for c in df_out.columns if c not in base_cols]
    df_out = df_out[base_cols + exog_cols]

    print(f"[Output sample]\n{df_out.tail(10)}")

    # ✅ เขียน market.parquet
    sdf = spark.createDataFrame(df_out)
    (
        sdf.repartition(1)
        .write.mode("overwrite")
        .parquet(OUT_PATH)
    )
    print(f"✅ Wrote market parquet to: {OUT_PATH}")

    # ✅ เขียน exog.parquet แยกไฟล์ (ถ้ามี)
    if exog_wide is not None:
        sdf_exog = spark.createDataFrame(exog_wide)
        (
            sdf_exog.repartition(1)
            .write.mode("overwrite")
            .parquet(EXOG_OUT_PATH)
        )
        print(f"✅ Wrote exog parquet to: {EXOG_OUT_PATH}")

    spark.stop()


if __name__ == "__main__":
    main()
