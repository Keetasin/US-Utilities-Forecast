#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lstm_run_tensorboard_batch_mainlike_strict.py

- ให้ผลลัพธ์ LSTM ใกล้ "โค้ดหลัก" สูงสุด
- Target = pct_change() เหมือนหลัก (ไม่ใช้ log-return เป็น target หลัก)
- สเกลทุกฟีเจอร์ (รวม target_ret ที่คอลัมน์ 0) ด้วย StandardScaler เดียว และ inverse ผ่าน scaler เดิม
- โหมด 'auto': H>63 => direct, ไม่งั้น recursive (ตรงหลัก)
- Recursive: พยากรณ์ทีละวัน, อัปเดต window ด้วย z ของ target + z ของ exog วันอนาคตทุกวัน
- Direct (ITERATIVE BLOCKS): พยากรณ์บล็อก log1p(ret) -> inverse -> กระจายรายวัน
  แล้ว "ต่อวัน" แปลง ret→z_target และ push เข้าหน้าต่างเหมือนเดินเวลา (พร้อมอัด exog z ของวันนั้น)
- Exog อนาคต: จำลองราคาแบบ random-walk จากสถิติ returns ย้อนหลัง -> คิดเป็น returns -> ทำ z ด้วย scaler_X
- เก็บ metrics + inference time + TensorBoard scalars ครบ

Artifacts: loss.png, history.csv, preds.csv, metrics.json, bt.png
สรุป: summary.csv (รวมเวลาพยากรณ์)

#Runfile
python lstm_run_tensorboard_batch_final.py --tickers AEP DUK SO ED EXC --steps 7 --epochs 60 --val_split 0.2
"""

from __future__ import annotations
import os, json, csv, argparse, uuid, random, sys, time, traceback
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler as SK_StandardScaler
import ta

# ===== Deterministic =====
random.seed(42); np.random.seed(42); tf.random.set_seed(42)
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# ===== Calendar <-> Business days (เหมือนโค้ดหลัก) =====
CALENDAR_TO_BDAYS = {7: 5, 180: 126, 365: 252}
def steps_to_bdays(steps: int) -> int:
    steps = int(steps)
    return CALENDAR_TO_BDAYS.get(steps, steps)

def to_bday_future_index(last_dt: pd.Timestamp, steps_b: int) -> pd.DatetimeIndex:
    start = pd.Timestamp(last_dt) + pd.offsets.BDay(1)
    return pd.bdate_range(start=start, periods=int(steps_b))

# ===== Exogenous config (ตามหลัก) =====
EXOG_TICKERS = {"oil": "CL=F", "gas": "NG=F", "xlu": "XLU"}

def get_exogenous(period="5y") -> pd.DataFrame:
    exog_df = pd.DataFrame()
    for name, tkr in EXOG_TICKERS.items():
        try:
            df = yf.download(tkr, period=period, progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            s = df["Close"].dropna()
            s = ensure_datetime_freq(s)
            exog_df[name] = s
        except Exception as e:
            print(f"[Exog] Failed to fetch {tkr}: {e}")
    return exog_df

def forecast_exog_series(exog_series: pd.Series, steps: int) -> pd.Series:
    try:
        returns = exog_series.pct_change().dropna()
        mu, sigma = returns.mean(), returns.std()
        last_val = float(exog_series.iloc[-1])
        fut = []
        H = steps_to_bdays(steps)
        for _ in range(H):
            shock = np.random.normal(mu, sigma)
            last_val *= (1.0 + shock)
            fut.append(last_val)
        return pd.Series(fut, index=to_bday_future_index(exog_series.index[-1], H))
    except Exception as e:
        print(f"[Exog Forecast Error] {e}")
        last_val = float(exog_series.iloc[-1])
        H = steps_to_bdays(steps)
        return pd.Series([last_val] * H, index=to_bday_future_index(exog_series.index[-1], H))

# ===== Helpers =====
def _as_series(x) -> pd.Series:
    if isinstance(x, pd.Series): s = x.copy()
    elif isinstance(x, pd.DataFrame):
        if "Close" in x.columns: s = x["Close"].copy()
        else:
            num_cols = [c for c in x.columns if np.issubdtype(x[c].dtype, np.number)]
            s = x[num_cols[0]].copy() if num_cols else x.iloc[:, 0].copy()
    else: s = pd.Series(x)
    s = pd.to_numeric(s, errors="coerce").dropna()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    return s

def ensure_datetime_freq(series: pd.Series | pd.DataFrame, use_bdays=True):
    s = series.copy()
    if not isinstance(s.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(s.index, errors="coerce")
    s = s[~s.index.isna()].sort_index()
    if len(s) == 0: return s
    if pd.infer_freq(s.index) is None:
        full_idx = pd.bdate_range(s.index.min(), s.index.max()) if use_bdays else pd.date_range(s.index.min(), s.index.max())
        s = s.reindex(full_idx).ffill()
    return s

def _prep_exog_rets(exog, target_index: pd.DatetimeIndex):
    if exog is None: return None
    if isinstance(exog, pd.Series): exog = exog.to_frame(name="ex0")
    if not isinstance(exog, pd.DataFrame) or exog.empty: return None
    ex_clean = ensure_datetime_freq(exog)
    ex_rets = ex_clean.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")
    ex_rets = ex_rets.reindex(target_index).ffill().dropna(how="all")
    if ex_rets is None or ex_rets.empty: return None
    ex_rets.columns = [f"ex_{c}" for c in ex_rets.columns]
    return ex_rets

def _to_2d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 1:
        return a.reshape(-1, 1)
    return a

# ===== Per-ticker tuned defaults (คง logicใกล้หลัก) =====
def tuned_params_for(ticker: str, steps: int):
    t = ticker.upper()
    H = steps_to_bdays(steps)
    hp = dict(
        lookback = 150 if H <= 130 else 180,
        clip_ret = 0.08,
        blend_drift = 0.20,
        agg_step = 5,     # ใช้เฉพาะ direct
    )
    if steps >= 360:
        if t == "SO": hp.update(lookback=200)
        elif t == "ED": hp.update(lookback=180)
    return hp

# ===== LSTM Core : main-like + strict future exog feeding =====
def lstm_forecast_v2(
    series, exog=None, steps=7, lookback=120, epochs=70, batch_size=64, patience=10,
    horizon_mode="auto", agg_step=5,
    clip_ret=0.08,         # match main
    blend_drift=0.20,      # match main
    return_history=False, tb_logdir=None, val_split=0.2
):
    """
    ส่งคืน 'ret ต่อวัน' ยาว H (business days)
      - recursive: ทำนาย z(target_ret) -> inverse ผ่าน scaler_X -> blend/clip -> push z + exog z วันถัดไป
      - direct (ITERATIVE BLOCKS): ทำนายบล็อก log1p(ret), inverse, แจกแจงรายวัน,
        แล้ว "ต่อวัน" คำนวณ z(target_ret) และ push เข้าหน้าต่าง เหมือนเดินเวลา พร้อมอัด exog z
    """
    s = ensure_datetime_freq(_as_series(series)).astype(np.float32)
    H = steps_to_bdays(int(steps))

    def _return(preds, hist=None):
        if return_history: return np.asarray(preds, np.float32), (hist or {})
        return np.asarray(preds, np.float32)

    if len(s) < lookback + 32:
        return _return(np.full(H, 0.0, np.float32), {})

    # ===== target = pct_return =====
    rets = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if isinstance(rets, pd.DataFrame): rets = rets.iloc[:, 0]
    drift = float(np.nanmean(rets.values[-60:])) if len(rets) >= 60 else float(np.nanmean(rets.values))

    # ===== indicators =====
    ind = pd.DataFrame(index=s.index)
    ind["rsi_14"] = ta.momentum.RSIIndicator(s, window=14).rsi()
    ind["ema_10"] = ta.trend.EMAIndicator(s, window=10).ema_indicator()
    ind["ema_50"] = ta.trend.EMAIndicator(s, window=50).ema_indicator()
    ind["macd"]   = ta.trend.MACD(s).macd()
    ind["bb_b"]   = ta.volatility.BollingerBands(s, window=20).bollinger_pband()
    ind["vol_20"] = s.pct_change().rolling(20).std()

    feats = [rets.rename("target_ret"), ind]
    ex_rets_hist = _prep_exog_rets(exog, rets.index)
    if ex_rets_hist is not None: feats.append(ex_rets_hist)

    df = pd.concat(feats, axis=1).dropna(how="any")
    if isinstance(df, pd.Series): df = df.to_frame(name=getattr(df, "name", "target_ret"))
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    df = df[num_cols]
    if df.shape[1] == 0 or len(df) == 0:
        return _return(np.full(H, float(np.nan_to_num(drift, nan=0.0)), np.float32), {})

    exog_cols = [c for c in df.columns if c.startswith("ex_")]
    exog_idx = [df.columns.get_loc(c) for c in exog_cols]
    n_exog = len(exog_idx)

    data = _to_2d(df.values.astype(np.float32))

    scaler_X = SK_StandardScaler()
    data_z = _to_2d(scaler_X.fit_transform(data))
    nfeat = data_z.shape[1]
    mean0 = float(scaler_X.mean_[0]); scale0 = float(scaler_X.scale_[0]) + 1e-8

    L = int(lookback)
    if len(data_z) <= L + 1:
        return _return(np.full(H, drift, np.float32), {})

    # เตรียม exog อนาคต → returns → z (ตาม scaler_X)
    exog_future_z = None
    fut_idx = to_bday_future_index(df.index[-1], H)
    if n_exog > 0 and exog is not None and isinstance(exog, pd.DataFrame) and not exog.empty:
        exog_hist_price = ensure_datetime_freq(exog).reindex(df.index).ffill().bfill()
        exog_future_price = pd.DataFrame(index=fut_idx)
        for col in exog_hist_price.columns:
            exog_future_price[col] = forecast_exog_series(exog_hist_price[col], H)
        exog_future_ret = exog_future_price.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
        exog_future_ret.columns = [f"ex_{c}" for c in exog_future_ret.columns]
        exog_future_ret = exog_future_ret.reindex(columns=exog_cols).fillna(0.0)

        exog_future_z = np.zeros((H, n_exog), dtype=np.float32)
        for j, col in enumerate(exog_cols):
            col_idx = df.columns.get_loc(col)
            mu = float(scaler_X.mean_[col_idx])
            sd = float(scaler_X.scale_[col_idx]) + 1e-8
            vals = exog_future_ret[col].to_numpy(dtype=np.float32, copy=True)
            exog_future_z[:, j] = (vals - mu) / sd

    # โหมด (เหมือนหลัก)
    if horizon_mode == "auto":
        mode = "direct" if H > 63 else "recursive"
    else:
        mode = horizon_mode

    # ===== โมเดล BiLSTM + Linear Dense (เหมือนหลัก) =====
    def _build_model(n_features: int) -> tf.keras.Model:
        from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Input
        from tensorflow.keras.models import Sequential
        m = Sequential([
            Input(shape=(L, n_features)),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dense(1)  # linear
        ])
        m.compile(optimizer="adam", loss=tf.keras.losses.Huber())
        return m

    # ===== TB callbacks =====
    class TBScalarLogger(tf.keras.callbacks.Callback):
        def __init__(self, writer): super().__init__(); self.writer = writer
        def _get_lr(self):
            try:
                if hasattr(self.model.optimizer, "_decayed_lr"):
                    return float(self.model.optimizer._decayed_lr(tf.float32).numpy())
                return float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            except Exception: return None
        def on_epoch_end(self, epoch, logs=None):
            if not self.writer: return
            logs = logs or {}
            with self.writer.as_default():
                if "loss" in logs: tf.summary.scalar("loss/train", logs["loss"], step=epoch)
                if "val_loss" in logs and logs["val_loss"] is not None: tf.summary.scalar("loss/val", logs["val_loss"], step=epoch)
                lr = self._get_lr()
                if lr is not None: tf.summary.scalar("learning_rate", lr, step=epoch)

    writer = tf.summary.create_file_writer(tb_logdir) if tb_logdir else None
    if writer is not None:
        with writer.as_default():
            tf.summary.text(
                "run_info",
                tf.convert_to_tensor(
                    f"ticker={os.environ.get('TB_TICKER','')}, steps={steps}, lookback={lookback}, "
                    f"epochs={epochs}, mode={horizon_mode}, clip_ret={clip_ret}, drift={drift:.6f}, "
                    f"notes={os.environ.get('TB_NOTES','')}"
                ),
                step=0
            )

    has_val = (val_split is not None and float(val_split) > 0)
    mon = "val_loss" if has_val else "loss"

    def _make_cbs():
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor=mon, patience=int(patience), restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor=mon, factor=0.5, patience=max(3, int(patience)//3), min_lr=1e-5),
            tf.keras.callbacks.TerminateOnNaN(),
        ]
        if writer is not None: cbs.append(TBScalarLogger(writer))
        return cbs

    history_dict = None

    # ======= recursive =======
    if mode == "recursive":
        X, y = [], []
        for i in range(L, len(data_z)):
            X.append(data_z[i - L:i])
            y.append(data_z[i, 0])  # z(target_ret)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        if X.size == 0:
            return _return(np.full(H, drift, np.float32), {})

        tf.keras.backend.clear_session()
        model = _build_model(nfeat)
        hist = model.fit(
            X, y,
            epochs=int(epochs), batch_size=int(batch_size), verbose=0,
            callbacks=_make_cbs(),
            validation_split=float(val_split) if has_val else 0.0,
            shuffle=False
        )
        history_dict = hist.history

        last_win = data_z[-L:].copy()
        preds_ret = []
        for t in range(H):
            # ใส่ exog z ของ "วัน t" ในแถวท้ายก่อนทำนาย
            if exog_future_z is not None:
                for j, col_pos in enumerate(exog_idx):
                    last_win[-1, col_pos] = exog_future_z[t, j]

            z_pred = float(model.predict(last_win[np.newaxis, :, :], verbose=0)[0, 0])

            # inverse -> ret_pred
            inv_vec = np.zeros((1, nfeat), dtype=np.float32)
            inv_vec[0, 0] = z_pred
            # inverse: z -> x = z*scale + mean
            ret_pred = float(inv_vec[0, 0] * scale0 + mean0)

            # blend + clip
            if blend_drift > 0:
                w = blend_drift * (t + 1) / H
                ret_pred = (1 - w) * ret_pred + w * drift
            if clip_ret is not None:
                ret_pred = float(np.clip(ret_pred, -clip_ret, clip_ret))

            preds_ret.append(ret_pred)

            # push target z ของวัน t (จาก ret_pred)
            z_target_today = (ret_pred - mean0) / scale0
            last_win = np.roll(last_win, -1, axis=0)
            last_win[-1, 0] = z_target_today
            # เตรียม exog z ของวันถัดไปในแถวใหม่ทันที
            if exog_future_z is not None:
                next_t = min(t + 1, H - 1)
                for j, col_pos in enumerate(exog_idx):
                    last_win[-1, col_pos] = exog_future_z[next_t, j]
            else:
                if nfeat > 1:
                    last_win[-1, 1:] = last_win[-2, 1:]

        return _return(np.asarray(preds_ret, np.float32), history_dict)

    # ======= direct (ITERATIVE BLOCKS) =======
    import math
    K = int(math.ceil(H / int(agg_step)))
    target_lr = np.log1p(df.iloc[:, 0].astype(np.float32)).values  # log1p(ret)

    max_shift = K * int(agg_step)
    if len(data_z) < (L + max_shift + 10):
        # ข้อมูลไม่พอ → fallback recursive
        return lstm_forecast_v2(
            series, exog, steps, lookback, epochs, batch_size, patience,
            horizon_mode="recursive", agg_step=agg_step,
            clip_ret=clip_ret, blend_drift=blend_drift,
            return_history=return_history, tb_logdir=tb_logdir, val_split=val_split
        )

    # สร้างชุด train blocks
    X_list, Y_list = [], []
    for i in range(L, len(data_z) - max_shift):
        X_list.append(data_z[i - L:i])
        yk = []
        for k in range(K):
            start = i + k * int(agg_step)
            end   = start + int(agg_step)
            yk.append(target_lr[start:end].sum())
        Y_list.append(yk)
    X = np.asarray(X_list, dtype=np.float32)
    Y = np.asarray(Y_list, dtype=np.float32)

    # scale Y (บล็อก)
    y_scaler = SK_StandardScaler()
    Y_z = y_scaler.fit_transform(Y)

    tf.keras.backend.clear_session()
    model = _build_model(nfeat)
    hist = model.fit(
        X, Y_z,
        epochs=int(epochs), batch_size=int(batch_size), verbose=0,
        callbacks=_make_cbs(),
        validation_split=float(val_split) if has_val else 0.0,
        shuffle=False
    )
    history_dict = hist.history

    # เริ่มจากหน้าต่างสุดท้าย แล้วเดินเวลาแบบ "จริง": รายวัน
    last_win = data_z[-L:].copy()
    preds_all = []

    day_ptr = 0
    for b in range(K):
        # อัปเดต exog z ของ "วันแรกในบล็อกนี้" ในแถวท้ายก่อน predict บล็อก
        if exog_future_z is not None:
            if day_ptr < H:
                for j, col_pos in enumerate(exog_idx):
                    last_win[-1, col_pos] = exog_future_z[day_ptr, j]

        # ทำนายบล็อก log1p-sum
        y_pred_z = model.predict(last_win[np.newaxis, :, :], verbose=0)[0]
        block_lr = float(y_scaler.inverse_transform(y_pred_z.reshape(1, -1))[0, b])

        # แจกแจงวันในบล็อกนี้
        days = int(agg_step) if b < K - 1 else (H - int(agg_step) * (K - 1))
        per_day_lr = block_lr / max(1, days)
        per_day_ret = float(np.expm1(per_day_lr))

        for d in range(days):
            # blend + clip วันต่อวัน
            t = day_ptr
            r = per_day_ret
            if clip_ret is not None:
                r = float(np.clip(r, -clip_ret, clip_ret))
            if blend_drift > 0:
                w = blend_drift * (t + 1) / H
                r = (1 - w) * r + w * drift
            preds_all.append(r)

            # คำนวณ z ของ target วันนี้แล้ว push
            z_today = (r - mean0) / scale0
            last_win = np.roll(last_win, -1, axis=0)
            last_win[-1, 0] = z_today

            # ใส่ exog z ของ "วันถัดไป" เข้าแถวท้ายทันที
            if exog_future_z is not None:
                next_t = min(t + 1, H - 1)
                for j, col_pos in enumerate(exog_idx):
                    last_win[-1, col_pos] = exog_future_z[next_t, j]
            else:
                if nfeat > 1:
                    last_win[-1, 1:] = last_win[-2, 1:]

            day_ptr += 1
            if day_ptr >= H:
                break
        if day_ptr >= H:
            break

    preds = np.asarray(preds_all[:H], dtype=np.float32)
    return _return(preds, history_dict)

# ===== Plot & CSV =====
def plot_history(history, out_png, title="Training Curve"):
    if not history: return
    loss = history.get("loss", [])
    val_loss = history.get("val_loss", [])
    plt.figure(figsize=(10, 5))
    if val_loss: plt.plot(range(len(val_loss)), val_loss, label="val_loss")
    if loss: plt.plot(range(len(loss)), loss, label="train_loss", alpha=0.5, linestyle="--")
    plt.title(title); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()

def plot_backtest(true_series, pred_series, out_png, title="Backtest"):
    plt.figure(figsize=(10,5))
    plt.plot(true_series.index, true_series.values, label="true")
    plt.plot(pred_series.index, pred_series.values, label="pred", linestyle="--")
    plt.title(title); plt.xlabel("date"); plt.ylabel("price"); plt.legend(); plt.tight_layout()
    plt.savefig(out_png, dpi=130); plt.close()

def append_summary(csv_path, row):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.exists(csv_path)
    headers = [
        "timestamp_utc", "batch_id", "run_id", "ticker", "steps",
        "MAE$", "RMSE$", "sMAPE%", "MAPE%", "MAE_ret", "DirAcc%", "MAE_norm%",
        "Infer_ms_future", "Infer_ms_future_per_step",
        "Infer_ms_backtest", "Infer_ms_backtest_per_step",
        "epochs", "lookback", "batch_size", "notes", "mode", "val_split"
    ]
    safe = {h: row.get(h, "") for h in headers}
    if not safe.get("timestamp_utc"):
        safe["timestamp_utc"] = datetime.utcnow().isoformat(timespec="seconds")
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not exists: w.writeheader()
        w.writerow(safe)

# ===== Run one ticker =====
def run_one(ticker, args, batch_id, batch_dir):
    run_id = f"{ticker}_{args.steps}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(batch_dir, run_id); os.makedirs(run_dir, exist_ok=True)

    # --- โครงสร้าง logDir: runs/tb/<batch>/<ticker>/S<steps>/<mode>/<run_id> ---
    tb_dir = os.path.join(args.logdir, "tb", batch_id, ticker, f"S{args.steps}", args.horizon_mode, run_id)
    os.makedirs(tb_dir, exist_ok=True)

    # context สำหรับ TB summary
    os.environ["TB_TICKER"] = ticker
    os.environ["TB_NOTES"] = args.notes or ""

    tuned = tuned_params_for(ticker, args.steps)
    lookback = args.lookback if args.lookback is not None else tuned["lookback"]
    H = steps_to_bdays(args.steps)
    default_clip = 0.07 if H <= 63 else 0.08
    clip_ret = default_clip if args.clip_ret is None else float(args.clip_ret)
    blend_drift = args.blend_drift if args.blend_drift is not None else tuned["blend_drift"]
    agg_step = args.agg_step if args.agg_step is not None else tuned["agg_step"]

    df = yf.download(ticker, period=args.period, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    s = ensure_datetime_freq(df["Close"].dropna())

    # ดึง exog price series ตาม period เดียวกับหลัก
    exog_all = get_exogenous(period=args.period)
    exog_all = ensure_datetime_freq(exog_all).reindex(s.index).ffill().bfill()

    # ==== อนาคต: inference time ====
    t0 = time.perf_counter()
    preds_ret_future, history = lstm_forecast_v2(
        s, exog=exog_all, steps=args.steps, lookback=lookback, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience,
        horizon_mode=args.horizon_mode, return_history=True,
        tb_logdir=tb_dir, val_split=args.val_split,
        agg_step=agg_step, clip_ret=clip_ret, blend_drift=blend_drift
    )
    infer_ms_future = (time.perf_counter() - t0) * 1000.0
    infer_ms_future_per_step = infer_ms_future / max(1, H)

    # สร้างราคาอนาคตจาก ret
    last_price = float(s.iloc[-1])
    vals, cur = [], last_price
    for r in preds_ret_future:
        cur *= (1.0 + float(r))
        vals.append(cur)
    idx_future = to_bday_future_index(s.index[-1], H)
    pred_series_future = pd.Series(np.array(vals, np.float32), index=idx_future)

    # Backtest (กันรั่วด้วย exog_all.iloc[:-H])
    if len(s) <= H + 10:
        raise RuntimeError(f"Not enough history for backtest (len={len(s)}, need > {H}+10)")
    train_s = s.iloc[:-H]; true_future = s.iloc[-H:]

    t1 = time.perf_counter()
    bt_ret = lstm_forecast_v2(
        train_s, exog=exog_all.iloc[:-H], steps=args.steps, lookback=lookback, epochs=args.epochs,
        batch_size=args.batch_size, patience=args.patience,
        horizon_mode=args.horizon_mode, return_history=False,
        tb_logdir=None, val_split=args.val_split,
        agg_step=agg_step, clip_ret=clip_ret, blend_drift=blend_drift
    )
    infer_ms_backtest = (time.perf_counter() - t1) * 1000.0
    infer_ms_backtest_per_step = infer_ms_backtest / max(1, H)

    bt_vals, curp = [], float(train_s.iloc[-1])
    for r in bt_ret:
        curp *= (1.0 + float(r))
        bt_vals.append(curp)
    bt_pred_series = pd.Series(np.array(bt_vals, np.float32), index=true_future.index)

    # ===== Metrics =====
    yt = true_future.to_numpy(np.float64).reshape(-1)
    yp = bt_pred_series.to_numpy(np.float64).reshape(-1)
    n = min(len(yt), len(yp)); yt = yt[:n]; yp = yp[:n]

    mae_price = float(np.mean(np.abs(yt - yp))) if n > 0 else float("nan")
    rmse_price = float(np.sqrt(np.mean((yt - yp) ** 2))) if n > 0 else float("nan")
    smape = float(2.0 * np.mean(np.abs(yp - yt) / (np.abs(yp) + np.abs(yt) + 1e-8)) * 100.0) if n > 0 else float("nan")
    mape_pct = float(np.mean(np.abs((yp - yt) / (yt + 1e-8))) * 100.0) if n > 0 else float("nan")

    def _ret(a):
        a = np.asarray(a).reshape(-1)
        return (a[1:] - a[:-1]) / (a[:-1] + 1e-8) if a.size >= 2 else np.array([], dtype=np.float64)
    rt = _ret(yt); rp = _ret(yp)
    m = min(len(rt), len(rp))
    mae_ret = float(np.mean(np.abs(rt[:m] - rp[:m]))) if m > 0 else float("nan")
    diracc = float((np.sign(rt[:m]) == np.sign(rp[:m])).mean() * 100.0) if m > 0 else float("nan")

    base = float(train_s.iloc[-1])
    yt_norm = yt / (base + 1e-8); yp_norm = yp / (base + 1e-8)
    mae_norm_pct = float(np.mean(np.abs(yt_norm - yp_norm)) * 100.0) if n > 0 else float("nan")

    print(
        f"[{ticker}] METRICS  "
        f"MAE$={mae_price:.4f}  RMSE$={rmse_price:.4f}  "
        f"sMAPE%={smape:.3f}  MAPE%={mape_pct:.3f}  "
        f"MAE_ret={mae_ret:.5f}  DirAcc%={diracc:.2f}  "
        f"MAE_norm%={mae_norm_pct:.3f}"
    )
    print(
        f"[{ticker}] INFER "
        f"future={infer_ms_future:.2f} ms ({infer_ms_future_per_step:.2f} ms/step),  "
        f"backtest={infer_ms_backtest:.2f} ms ({infer_ms_backtest_per_step:.2f} ms/step)"
    )

    # Artifacts
    plot_history(history, os.path.join(run_dir, f"{ticker}_{args.steps}_loss.png"), title=f"{ticker} Training Curve")
    plot_backtest(true_future, bt_pred_series,
                  os.path.join(run_dir, f"{ticker}_{args.steps}_bt.png"),
                  title=f"{ticker} Backtest ({args.steps} cal days)")
    if history: pd.DataFrame(history).to_csv(os.path.join(run_dir, "history.csv"), index=False)
    pred_series_future.to_csv(os.path.join(run_dir, "preds.csv"), header=["forecast"])
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "MAE$": mae_price, "RMSE$": rmse_price,
            "sMAPE%": smape, "MAPE%": mape_pct,
            "MAE_ret": mae_ret, "DirAcc%": diracc,
            "MAE_norm%": mae_norm_pct,
            "Infer_ms_future": infer_ms_future,
            "Infer_ms_future_per_step": infer_ms_future_per_step,
            "Infer_ms_backtest": infer_ms_backtest,
            "Infer_ms_backtest_per_step": infer_ms_backtest_per_step
        }, f, indent=2)

    row = {
        "batch_id": batch_id, "run_id": run_id, "ticker": ticker, "steps": args.steps,
        "MAE$": mae_price, "RMSE$": rmse_price, "sMAPE%": smape, "MAPE%": mape_pct,
        "MAE_ret": mae_ret, "DirAcc%": diracc, "MAE_norm%": mae_norm_pct,
        "Infer_ms_future": infer_ms_future,
        "Infer_ms_future_per_step": infer_ms_future_per_step,
        "Infer_ms_backtest": infer_ms_backtest,
        "Infer_ms_backtest_per_step": infer_ms_backtest_per_step,
        "epochs": args.epochs, "lookback": lookback, "batch_size": args.batch_size,
        "notes": args.notes, "mode": args.horizon_mode, "val_split": args.val_split
    }
    append_summary(os.path.join(batch_dir, "summary.csv"), row)
    append_summary(os.path.join(args.logdir, "summary.csv"), row)
    print(f"[{ticker}] done -> {run_dir}")

    # TensorBoard: inference times
    try:
        writer_tb = tf.summary.create_file_writer(tb_dir)
        with writer_tb.as_default():
            tf.summary.scalar("inference_ms/future", infer_ms_future, step=0)
            tf.summary.scalar("inference_ms/future_per_step", infer_ms_future_per_step, step=0)
            tf.summary.scalar("inference_ms/backtest", infer_ms_backtest, step=0)
            tf.summary.scalar("inference_ms/backtest_per_step", infer_ms_backtest_per_step, step=0)
    except Exception:
        pass

    return run_dir

# ===== Main =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", nargs="+", required=True, help="เช่น --tickers AEP DUK SO ED EXC")
    ap.add_argument("--steps", type=int, default=365)
    ap.add_argument("--period", type=str, default="10y")
    ap.add_argument("--lookback", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=70)     # ให้ตรงหลัก
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--horizon_mode", type=str, default="auto", choices=["auto", "recursive", "direct"])
    ap.add_argument("--val_split", type=float, default=0.2)

    # tuned knobs
    ap.add_argument("--agg_step", type=int, default=None)
    ap.add_argument("--clip_ret", type=float, default=None)  # override ตาม H ภายหลัง
    ap.add_argument("--blend_drift", type=float, default=None)

    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--logdir", type=str, default="runs")
    args = ap.parse_args()

    ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    batch_id = f"batch_{ts}_{uuid.uuid4().hex[:6]}"
    batch_dir = os.path.join(args.logdir, batch_id)
    os.makedirs(batch_dir, exist_ok=True)

    print(f"[BATCH] {batch_id}")
    for tk in args.tickers:
        try:
            run_one(tk, args, batch_id, batch_dir)
        except Exception as e:
            print(f"[{tk}] ERROR: {e}\n{traceback.format_exc()}")

    print("\nเปิด TensorBoard:\n  tensorboard --logdir runs/tb\nแล้วเปิด http://localhost:6006")

if __name__ == "__main__":
    if sys.platform.startswith("win") and sys.version_info >= (3, 13):
        raise SystemExit("ERROR: ใช้ Python 3.12 ใน virtualenv สำหรับ TensorFlow บน Windows")
    main()