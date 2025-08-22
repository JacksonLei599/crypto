{\rtf1\ansi\ansicpg936\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 from __future__ import annotations\
import os\
import sys\
import time\
import math\
import logging\
from dataclasses import dataclass\
from typing import Optional, Tuple\
\
import numpy as np\
import pandas as pd\
import ccxt  # type: ignore\
\
# ---------------------------- CONFIG ---------------------------------\
@dataclass\
class Config:\
    exchange_id: str = os.getenv("EXCHANGE_ID", "binance")  # e.g., 'binance','bybit','okx','kraken'\
    symbol: str = os.getenv("SYMBOL", "BTC/USDT")\
    timeframe: str = os.getenv("TIMEFRAME", "1h")  # e.g., '1m','5m','15m','1h','4h','1d'\
    lookback_bars: int = int(os.getenv("LOOKBACK_BARS", "1000"))  # for fetch OHLCV\
\
    # Strategy params\
    ema_fast: int = int(os.getenv("EMA_FAST", "20"))\
    ema_slow: int = int(os.getenv("EMA_SLOW", "50"))\
    rsi_period: int = int(os.getenv("RSI_PERIOD", "14"))\
    rsi_buy_threshold: float = float(os.getenv("RSI_BUY", "50"))\
    rsi_sell_threshold: float = float(os.getenv("RSI_SELL", "50"))\
\
    # Backtest costs\
    taker_fee: float = float(os.getenv("TAKER_FEE", "0.001"))   # 0.1%\
    slippage_bps: float = float(os.getenv("SLIPPAGE_BPS", "2"))  # 2 bps = 0.02%\
\
    # Portfolio\
    init_cash: float = float(os.getenv("INIT_CASH", "10000"))\
    risk_per_trade: float = float(os.getenv("RISK_PER_TRADE", "1.0"))  # fraction of cash to deploy per entry (0..1)\
    max_position_value: float = float(os.getenv("MAX_POS_VAL", "1.0"))  # cap vs equity (0..1)\
\
    # Paper/Live\
    dry_run: bool = os.getenv("DRY_RUN", "true").lower() != "false"  # default True\
    poll_sec: int = int(os.getenv("POLL_SEC", "30"))\
\
CONFIG = Config()\
\
# ---------------------------- LOGGING --------------------------------\
logging.basicConfig(\
    level=logging.INFO,\
    format="%(asctime)s | %(levelname)s | %(message)s",\
)\
log = logging.getLogger("quant")\
\
# ---------------------------- HELPERS --------------------------------\
\
def load_exchange(cfg: Config) -> ccxt.Exchange:\
    ex_class = getattr(ccxt, cfg.exchange_id)\
    api_key = os.getenv("EXCHANGE_API_KEY", "")\
    secret = os.getenv("EXCHANGE_SECRET", "")\
    password = os.getenv("EXCHANGE_PASSWORD", "")\
    exchange = ex_class(\{\
        "apiKey": api_key or None,\
        "secret": secret or None,\
        "password": password or None,\
        "enableRateLimit": True,\
        "options": \{"defaultType": "spot"\},\
    \})\
    return exchange\
\
\
def fetch_ohlcv_df(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:\
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)\
    if not raw:\
        raise RuntimeError("No OHLCV returned. Check symbol/timeframe.")\
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])\\\
           .assign(timestamp=lambda d: pd.to_datetime(d["timestamp"], unit="ms", utc=True))\
    df = df.set_index("timestamp").sort_index()\
    return df\
\
\
def ema(series: pd.Series, period: int) -> pd.Series:\
    return series.ewm(span=period, adjust=False).mean()\
\
\
def rsi(series: pd.Series, period: int = 14) -> pd.Series:\
    delta = series.diff()\
    up = delta.clip(lower=0.0)\
    down = -delta.clip(upper=0.0)\
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()\
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()\
    rs = roll_up / (roll_down + 1e-12)\
    rsi_val = 100.0 - (100.0 / (1.0 + rs))\
    return rsi_val\
\
\
def apply_indicators(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:\
    df = df.copy()\
    df["ema_fast"] = ema(df["close"], cfg.ema_fast)\
    df["ema_slow"] = ema(df["close"], cfg.ema_slow)\
    df["rsi"] = rsi(df["close"], cfg.rsi_period)\
    return df\
\
\
def generate_signals(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:\
    df = df.copy()\
    df["bull"] = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] >= cfg.rsi_buy_threshold)\
    df["bear"] = (df["ema_fast"] < df["ema_slow"]) | (df["rsi"] < cfg.rsi_sell_threshold)\
    df["long_entry"] = df["bull"] & (~df["bull"].shift(1).fillna(False))\
    df["long_exit"] = df["bear"] & (~df["bear"].shift(1).fillna(False))\
    return df\
\
\
def backtest(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, dict]:\
    df = df.copy()\
    fee = cfg.taker_fee\
    slip = cfg.slippage_bps / 1e4\
\
    equity = cfg.init_cash\
    cash = cfg.init_cash\
    position = 0.0\
\
    eq_curve = []\
    pos_val_series = []\
\
    for ts, row in df.iterrows():\
        price = float(row["close"])\
        pos_val = position * price\
        equity = cash + pos_val\
        eq_curve.append((ts, equity))\
        pos_val_series.append((ts, pos_val))\
\
        if row.get("long_entry", False) and position <= 1e-12:\
            target_value = min(cfg.risk_per_trade * equity, cfg.max_position_value * equity)\
            buy_price = price * (1 + slip) * (1 + fee)\
            qty = max(0.0, (target_value) / buy_price)\
            cost = qty * buy_price\
            if cost <= cash and qty > 0:\
                cash -= cost\
                position += qty\
                log.info(f"BUY \{qty:.6f\} at ~\{buy_price:.2f\} on \{ts\}")\
\
        elif row.get("long_exit", False) and position > 1e-12:\
            sell_price = price * (1 - slip) * (1 - fee)\
            proceeds = position * sell_price\
            cash += proceeds\
            log.info(f"SELL \{position:.6f\} at ~\{sell_price:.2f\} on \{ts\}")\
            position = 0.0\
\
    if len(df) > 0:\
        price = float(df.iloc[-1]["close"])\
        equity = cash + position * price\
        eq_curve.append((df.index[-1], equity))\
\
    eq_df = pd.DataFrame(eq_curve, columns=["timestamp", "equity"]).set_index("timestamp")\
    pos_df = pd.DataFrame(pos_val_series, columns=["timestamp","pos_val"]).set_index("timestamp")\
    joined = df.join(eq_df).join(pos_df)\
\
    ret = joined["equity"].pct_change().fillna(0.0)\
    total_return = joined["equity"].iloc[-1] / joined["equity"].iloc[0] - 1.0\
\
    bars_per_year = _estimate_bars_per_year(CONFIG.timeframe)\
    sharpe = _sharpe(ret, bars_per_year)\
    mdd = _max_drawdown(joined["equity"])\
\
    stats = \{\
        "total_return": total_return,\
        "sharpe": sharpe,\
        "max_drawdown": mdd,\
        "final_equity": float(joined["equity"].iloc[-1]),\
    \}\
    return joined, stats\
\
\
def _estimate_bars_per_year(timeframe: str) -> int:\
    tf_map = \{\
        "1m": 60*24*365,\
        "5m": 12*24*365,\
        "15m": 4*24*365,\
        "1h": 24*365,\
        "4h": 6*365,\
        "1d": 365,\
    \}\
    return tf_map.get(timeframe, 24*365)\
\
\
def _sharpe(ret: pd.Series, bars_per_year: int, risk_free: float = 0.0) -> float:\
    excess = ret - (risk_free / bars_per_year)\
    mu = excess.mean()\
    sigma = excess.std(ddof=1)\
    if sigma <= 1e-12:\
        return 0.0\
    return float(math.sqrt(bars_per_year) * mu / sigma)\
\
\
def _max_drawdown(equity: pd.Series) -> float:\
    roll_max = equity.cummax()\
    drawdown = equity / roll_max - 1.0\
    return float(drawdown.min())\
\
# ------------------------- EXECUTION (PAPER) --------------------------\
class PaperTrader:\
    def __init__(self, exchange: ccxt.Exchange, cfg: Config):\
        self.exchange = exchange\
        self.cfg = cfg\
        self.position = 0.0\
        self.cash = cfg.init_cash\
\
    def run_once(self):\
        df = fetch_ohlcv_df(self.exchange, self.cfg.symbol, self.cfg.timeframe, self.cfg.lookback_bars)\
        df = apply_indicators(df, self.cfg)\
        df = generate_signals(df, self.cfg)\
        last = df.iloc[-1]\
        price = float(last["close"])\
\
        equity = self.cash + self.position * price\
        log.info(f"Equity: \{equity:.2f\} | Pos: \{self.position:.6f\} \{self.cfg.symbol.split('/')[0]\} | Px: \{price:.2f\}")\
\
        if last["long_entry"] and self.position <= 1e-12:\
            fee = self.cfg.taker_fee\
            slip = self.cfg.slippage_bps / 1e4\
            target_value = min(self.cfg.risk_per_trade * equity, self.cfg.max_position_value * equity)\
            buy_price = price * (1 + slip) * (1 + fee)\
            qty = max(0.0, target_value / buy_price)\
            cost = qty * buy_price\
            if cost <= self.cash and qty > 0:\
                self.cash -= cost\
                self.position += qty\
                log.info(f"[PAPER] BUY \{qty:.6f\} at ~\{buy_price:.2f\}")\
        elif last["long_exit"] and self.position > 1e-12:\
            fee = self.cfg.taker_fee\
            slip = self.cfg.slippage_bps / 1e4\
            sell_price = price * (1 - slip) * (1 - fee)\
            proceeds = self.position * sell_price\
            self.cash += proceeds\
            log.info(f"[PAPER] SELL \{self.position:.6f\} at ~\{sell_price:.2f\}")\
            self.position = 0.0\
\
# ------------------------------ CLI ----------------------------------\
\
def cmd_backtest(cfg: Config):\
    ex = load_exchange(cfg)\
    log.info(f"Fetching \{cfg.symbol\} \{cfg.timeframe\} \{cfg.lookback_bars\} bars from \{cfg.exchange_id\} ...")\
    df = fetch_ohlcv_df(ex, cfg.symbol, cfg.timeframe, cfg.lookback_bars)\
    df = apply_indicators(df, cfg)\
    df = generate_signals(df, cfg)\
    joined, stats = backtest(df, cfg)\
    log.info("Backtest stats: %s", stats)\
    print(joined[["close","ema_fast","ema_slow","rsi","long_entry","long_exit","equity"]].tail(10))\
\
\
def cmd_paper(cfg: Config):\
    ex = load_exchange(cfg)\
    trader = PaperTrader(ex, cfg)\
    log.info("Starting PAPER loop (Ctrl+C to stop)\'85")\
    try:\
        while True:\
            trader.run_once()\
            time.sleep(cfg.poll_sec)\
    except KeyboardInterrupt:\
        log.info("Stopped by user.")\
\
\
if __name__ == "__main__":\
    if len(sys.argv) < 2:\
        print("Usage: python main.py [backtest|paper]")\
        sys.exit(0)\
    cmd = sys.argv[1].lower()\
    if cmd == "backtest":\
        cmd_backtest(CONFIG)\
    elif cmd == "paper":\
        cmd_paper(CONFIG)\
    else:\
        print("Unknown command. Use: backtest | paper")}