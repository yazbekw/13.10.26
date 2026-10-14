"""
Binance Fusion Bot v4.1 — Single File
يشغّل: 6 عملات × (Kline 15m + aggTrade) + بث التصفيات العام
يرسل تنبيهات Telegram: إشارات اختراق (VPVR + Trendline) + شلالات تصفيات
"""

import asyncio, aiohttp, websockets, json, os, time
import numpy as np
from collections import deque
from sklearn.linear_model import RANSACRegressor
from scipy.signal import argrelextrema
from dotenv import load_dotenv

# ═══════════════════════════════════════════════════════════
# الإعدادات
# ═══════════════════════════════════════════════════════════
load_dotenv()
TG_TOKEN = os.getenv("TG_TOKEN")
TG_CHAT  = os.getenv("TG_CHAT")
if not TG_TOKEN or not TG_CHAT:
    raise SystemExit("❌ ضع TG_TOKEN و TG_CHAT في .env")

SYMBOLS         = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT"]
WINDOW_SEC      = 60

MIN_SCORE       = 50
SIGNAL_COOLDOWN = 900
LIQ_FRESH_SEC   = 300
LIQ_COOLDOWN    = 180

VPVR_LOOKBACK   = 5000
VPVR_BINS       = 50
VPVR_VA_PCT     = 0.70
VPVR_GRID_SEC   = 900
VPVR_CACHE_SEC  = 5.0

TREND_WINDOW    = 200
TREND_PIVOT     = 5
TREND_VOL_MULT  = 1.5
TREND_GAP       = 10
TREND_MAX_SLOPE = 0.001

LIQ_HISTORY     = 60
LIQ_MAD_THRESH  = 3.5
LIQ_MIN_VALUE   = 50_000

# ═══════════════════════════════════════════════════════════
# Telegram Alerter — Queue + Worker + Cooldown-on-Success
# ═══════════════════════════════════════════════════════════
class Alerter:
    def __init__(self):
        self.session = None
        self.queue = asyncio.Queue(maxsize=2000)
        self.worker = None
        self.cooldowns = {}
        self.stopping = False

    async def start(self):
        self.session = aiohttp.ClientSession()
        self.worker = asyncio.create_task(self._worker())

    async def close(self):
        self.stopping = True
        if self.worker:
            self.worker.cancel()
            try: await self.worker
            except asyncio.CancelledError: pass
        if self.session: await self.session.close()

    async def _worker(self):
        while not self.stopping:
            try:
                key, msg, cd = await self.queue.get()
                if time.time() - self.cooldowns.get(key, 0) < cd:
                    self.queue.task_done(); continue
                if await self._send(msg):
                    self.cooldowns[key] = time.time()
                self.queue.task_done()
            except asyncio.CancelledError: break
            except Exception as e: print(f"⚠️ TG: {e}")

    async def _send(self, msg):
        url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
        try:
            async with self.session.post(url,
                json={"chat_id": TG_CHAT, "text": msg, "parse_mode": "HTML"},
                timeout=aiohttp.ClientTimeout(total=5)) as r:
                return r.status == 200
        except Exception as e:
            print(f"⚠️ TG send: {e}"); return False

    def notify(self, key, msg, cooldown=300):
        try: self.queue.put_nowait((key, msg, cooldown))
        except asyncio.QueueFull: pass

# ═══════════════════════════════════════════════════════════
# VPVR — Fixed Price Grid
# ═══════════════════════════════════════════════════════════
class VPVR:
    def __init__(self):
        self.prices = deque(maxlen=VPVR_LOOKBACK)
        self.volumes = deque(maxlen=VPVR_LOOKBACK)
        self.edges = None; self.centers = None
        self._last_grid = 0; self._last_cache = 0; self._cached = None

    def add(self, price, volume):
        self.prices.append(price); self.volumes.append(price * volume)

    def _rebuild(self, now):
        if len(self.prices) < 100: return
        p = np.array(self.prices)
        lo, hi = p.min(), p.max()
        if hi <= lo: return
        pad = (hi - lo) * 0.01
        self.edges = np.linspace(lo - pad, hi + pad, VPVR_BINS + 1)
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        self._last_grid = now

    def value_area(self):
        now = time.time()
        if self._cached and now - self._last_cache < VPVR_CACHE_SEC:
            return self._cached
        if len(self.prices) < 500: return None
        if self.edges is None or now - self._last_grid >= VPVR_GRID_SEC:
            self._rebuild(now)
        if self.edges is None: return None
        self._last_cache = now

        p = np.array(self.prices); v = np.array(self.volumes)
        idx = np.clip(np.digitize(p, self.edges) - 1, 0, VPVR_BINS - 1)
        profile = np.zeros(VPVR_BINS)
        np.add.at(profile, idx, v)

        poc = int(profile.argmax()); total = profile.sum()
        if total <= 0: return None
        target = total * VPVR_VA_PCT
        inc, acc = {poc}, profile[poc]
        low, high = poc - 1, poc + 1
        while acc < target and (low >= 0 or high < VPVR_BINS):
            vb = profile[low] if low >= 0 else -1
            va = profile[high] if high < VPVR_BINS else -1
            if va >= vb: inc.add(high); acc += va; high += 1
            else:        inc.add(low);  acc += vb; low  -= 1

        self._cached = {
            "POC": float(self.centers[poc]),
            "VAH": float(self.centers[max(inc)]),
            "VAL": float(self.centers[min(inc)]),
        }
        return self._cached

# ═══════════════════════════════════════════════════════════
# Trendline — RANSAC + Slope + Touch Gap
# ═══════════════════════════════════════════════════════════
class Trendline:
    def __init__(self):
        self.prices = deque(maxlen=TREND_WINDOW)
        self.volumes = deque(maxlen=TREND_WINDOW)

    def add(self, close, volume):
        self.prices.append(close); self.volumes.append(volume)

    def _touches(self, pivots, y, s, b, res):
        hits = [p for p in pivots if abs(y[p] - (s * p + b)) < res]
        if not hits: return []
        kept = [hits[0]]
        for p in hits[1:]:
            if p - kept[-1] >= TREND_GAP: kept.append(p)
        return kept

    def detect(self):
        if len(self.prices) < 50: return None
        y = np.array(self.prices); v = np.array(self.volumes)
        cur = y[-1]; mean_y = np.mean(y)

        avg_vol = np.mean(v[-21:-1])
        if avg_vol <= 0: return None
        vr = v[-1] / avg_vol
        if vr < TREND_VOL_MULT: return None

        peaks   = argrelextrema(y, np.greater, order=TREND_PIVOT)[0]
        valleys = argrelextrema(y, np.less,    order=TREND_PIVOT)[0]
        if len(peaks) < 2 or len(valleys) < 2: return None

        out, x = [], len(y) - 1
        res = 0.005 * mean_y

        try:
            r = RANSACRegressor(min_samples=2, residual_threshold=res, random_state=42)
            r.fit(peaks.reshape(-1, 1), y[peaks])
            s, b = r.estimator_.coef_[0], r.estimator_.intercept_
            if s / mean_y <= TREND_MAX_SLOPE:
                lvl = s * x + b
                if cur > lvl:
                    t = self._touches(peaks, y, s, b, res * 2)
                    if len(t) >= 2:
                        out.append({"type": "UP", "level": float(lvl),
                                    "touches": len(t), "vol_ratio": float(vr)})
        except Exception: pass

        try:
            r = RANSACRegressor(min_samples=2, residual_threshold=res, random_state=42)
            r.fit(valleys.reshape(-1, 1), y[valleys])
            s, b = r.estimator_.coef_[0], r.estimator_.intercept_
            if s / mean_y >= -TREND_MAX_SLOPE:
                lvl = s * x + b
                if cur < lvl:
                    t = self._touches(valleys, y, s, b, res * 2)
                    if len(t) >= 2:
                        out.append({"type": "DOWN", "level": float(lvl),
                                    "touches": len(t), "vol_ratio": float(vr)})
        except Exception: pass

        return out or None

# ═══════════════════════════════════════════════════════════
# Liquidation Window — MAD + Correct Classification
# ═══════════════════════════════════════════════════════════
class LiqWindow:
    def __init__(self):
        self.history = deque(maxlen=LIQ_HISTORY)
        self.long = 0.0; self.short = 0.0; self.count = 0

    def add(self, value, side):
        if side == "SELL": self.long  += value
        else:              self.short += value
        self.count += 1

    def close(self):
        entry = {"long": self.long, "short": self.short,
                 "total": self.long + self.short, "ts": time.time()}
        self.history.append(entry)
        self.long = self.short = 0.0; self.count = 0
        return self._anomaly()

    def _anomaly(self):
        if len(self.history) < 15: return None
        base = np.array([h["total"] for h in list(self.history)[:-1]])
        med = np.median(base)
        mad = np.median(np.abs(base - med))
        rstd = 1.4826 * mad
        if rstd < 1e-6:
            rstd = base.std()
            if rstd < 1e-6: return None
        last = self.history[-1]
        if last["total"] < LIQ_MIN_VALUE: return None
        z = (last["total"] - med) / rstd
        if z < LIQ_MAD_THRESH: return None
        if last["short"] > last["long"] * 1.5:   kind = "SHORT_SQUEEZE"
        elif last["long"] > last["short"] * 1.5: kind = "LONG_SQUEEZE"
        else:                                     kind = "MIXED"
        return {"z": float(z), "last": last, "kind": kind}

# ═══════════════════════════════════════════════════════════
# Symbol State + Evaluator
# ═══════════════════════════════════════════════════════════
class State:
    def __init__(self, symbol):
        self.symbol = symbol
        self.vpvr = VPVR()
        self.trend = Trendline()
        self.liq = LiqWindow()
        self.last_sig = 0

def evaluate(st, sig, price):
    if time.time() - st.last_sig < SIGNAL_COOLDOWN: return None

    va = st.vpvr.value_area()
    if va is None: return None
    if sig["type"] == "UP"   and price <= va["VAH"]: return None
    if sig["type"] == "DOWN" and price >= va["VAL"]: return None

    if not st.liq.history: return None
    w = st.liq.history[-1]
    if time.time() - w["ts"] > LIQ_FRESH_SEC: return None

    score, reasons = 30, [f"إغلاق {sig['level']:.4f} | لمسات {sig['touches']}"]
    vr = sig["vol_ratio"]
    if   vr >= 2.5: score += 20; reasons.append(f"حجم {vr:.2f}x")
    elif vr >= 2.0: score += 12; reasons.append(f"حجم {vr:.2f}x")
    else:           score += 5;  reasons.append(f"حجم {vr:.2f}x")

    score += 20
    ref = va["VAH"] if sig["type"] == "UP" else va["VAL"]
    reasons.append(f"VPVR {ref:.4f}")

    if sig["type"] == "UP" and w["kind"] == "SHORT_SQUEEZE":
        score += 25; reasons.append(f"Short squeeze {w['short']:,.0f}")
    elif sig["type"] == "DOWN" and w["kind"] == "LONG_SQUEEZE":
        score += 25; reasons.append(f"Long squeeze {w['long']:,.0f}")

    dist = abs(price - sig["level"]) / price * 100
    if dist < 0.3: score += 5; reasons.append(f"نظيف {dist:.2f}%")

    if score < MIN_SCORE: return None
    st.last_sig = time.time()
    return {"score": score, "reasons": reasons}

# ═══════════════════════════════════════════════════════════
# Fusion Controller
# ═══════════════════════════════════════════════════════════
class Fusion:
    def __init__(self, alerter):
        self.alerter = alerter
        self.states = {s: State(s) for s in SYMBOLS}

    def trade(self, symbol, price, vol):
        st = self.states.get(symbol)
        if st: st.vpvr.add(price, vol)

    def candle(self, symbol, close, vol):
        st = self.states.get(symbol)
        if not st: return
        st.trend.add(close, vol)
        sigs = st.trend.detect()
        if not sigs: return
        for sig in sigs:
            r = evaluate(st, sig, close)
            if r:
                emoji = "🟢" if sig["type"] == "UP" else "🔴"
                msg = (f"{emoji} <b>{symbol}</b> — {sig['type']}\n"
                       f"السعر: {close:.4f}\n"
                       f"درجة التوافق: {r['score']}\n"
                       + "\n".join(f"• {x}" for x in r["reasons"]))
                self.alerter.notify(f"{symbol}_{sig['type']}", msg, cooldown=SIGNAL_COOLDOWN)

    def liq(self, symbol, value, side):
        st = self.states.get(symbol)
        if st: st.liq.add(value, side)

    async def window_loop(self):
        while True:
            await asyncio.sleep(WINDOW_SEC)
            for sym, st in self.states.items():
                a = st.liq.close()
                if not a: continue
                last, kind = a["last"], a["kind"]
                emoji = {"SHORT_SQUEEZE": "🔥", "LONG_SQUEEZE": "🩸", "MIXED": "⚠️"}[kind]
                msg = (f"🚨 <b>شلال تصفيات — {sym}</b>\n"
                       f"النوع: {kind} {emoji}\n"
                       f"Z: {a['z']:.2f}\n"
                       f"الإجمالي: {last['total']:,.0f}\n"
                       f"LONG: {last['long']:,.0f} | SHORT: {last['short']:,.0f}")
                self.alerter.notify(f"liq_{sym}", msg, cooldown=LIQ_COOLDOWN)

# ═══════════════════════════════════════════════════════════
# Binance Connector
# ═══════════════════════════════════════════════════════════
class Connector:
    def __init__(self, fusion):
        self.fusion = fusion
        self.stop_ev = asyncio.Event()
        self.tasks = []

    async def _loop(self, url, handler, name):
        backoff = 1
        while not self.stop_ev.is_set():
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    print(f"✅ {name}"); backoff = 1
                    async for raw in ws:
                        if self.stop_ev.is_set(): break
                        try: handler(json.loads(raw))
                        except Exception as e: print(f"⚠️ {name}: {e}")
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"⚠️ {name} down: {e}. retry {backoff}s")
                try: await asyncio.wait_for(self.stop_ev.wait(), timeout=backoff)
                except asyncio.TimeoutError: pass
                backoff = min(backoff * 2, 30)

    def _on_liq(self, d):
        if d.get('e') != 'forceOrder': return
        o = d['o']; s = o['s']
        if s not in SYMBOLS: return
        self.fusion.liq(s, float(o['p']) * float(o['q']), o['S'])

    def _on_kline(self, d):
        k = d['k']
        if not k['x']: return
        self.fusion.candle(k['s'], float(k['c']), float(k['v']))

    def _on_trade(self, d):
        self.fusion.trade(d['s'], float(d['p']), float(d['q']))

    async def run(self):
        self.tasks.append(asyncio.create_task(self._loop(
            "wss://fstream.binance.com/ws/!forceOrder@arr", self._on_liq, "liq")))
        for s in SYMBOLS:
            self.tasks.append(asyncio.create_task(self._loop(
                f"wss://fstream.binance.com/ws/{s.lower()}@kline_15m",
                self._on_kline, f"{s} 15m")))
            self.tasks.append(asyncio.create_task(self._loop(
                f"wss://fstream.binance.com/ws/{s.lower()}@aggTrade",
                self._on_trade, f"{s} trades")))
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def stop(self):
        print("🛑 إيقاف...")
        self.stop_ev.set()
        for t in self.tasks: t.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════
async def main():
    print("🚀 Binance Fusion Bot v4.1")
    alerter = Alerter(); await alerter.start()
    alerter.notify("startup", "✅ <b>البوت يعمل</b> — 6 عملات × 15m", cooldown=0)

    fusion = Fusion(alerter)
    conn = Connector(fusion)
    win = asyncio.create_task(fusion.window_loop())

    try:
        await conn.run()
    except KeyboardInterrupt:
        pass
    finally:
        win.cancel()
        await conn.stop()
        await alerter.close()
        print("✅ تم الإيقاف")

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
