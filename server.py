"""Latent Tape - backend.

Streams Kraken order books for several USD pairs, trains one small
autoencoder per pair online, and pushes weights / latent state to browsers.
"""
import asyncio, heapq, json
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

import aiohttp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse

SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "DOGE/USD"]
KRAKEN_URL = "wss://ws.kraken.com/v2"
BOOK_DEPTH = 100      # levels requested from Kraken
DEPTH = 10            # levels fed to the network (per side)
STEPS = 5             # gradient steps per tick, per pair
TICK = 0.12           # seconds between pushes to browsers
PRICE_SCALE = 1000.0  # lifts tiny price offsets to a learnable range
torch.set_num_threads(1)


class AutoEncoder(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.layer1 = nn.Linear(n, n // 2)
        self.layer2 = nn.Linear(n // 2, n // 4)
        self.layer3 = nn.Linear(n // 4, n // 2)
        self.layer4 = nn.Linear(n // 2, n)
        self.relu = nn.ReLU()

    def latent(self, x):
        return self.layer2(self.relu(self.layer1(x)))

    def forward(self, x):
        x = self.relu(self.layer3(self.relu(self.latent(x))))
        return self.layer4(x)

    def weights(self):
        return [l.weight.detach().numpy() for l in
                (self.layer1, self.layer2, self.layer3, self.layer4)]


def organize_book(bids, asks, depth=DEPTH):
    b = sorted(bids.items(), reverse=True)[:depth]
    a = sorted(asks.items())[:depth]
    if len(b) < depth or len(a) < depth:
        return None
    b, a = np.array(b), np.array(a)
    mid = (b[0, 0] + a[0, 0]) / 2.0
    nb, na = (b[:, 0] - mid) / mid, (a[:, 0] - mid) / mid
    vb, va = b[:, 1] / b[:, 1].sum(), a[:, 1] / a[:, 1].sum()
    x = np.concatenate([nb * PRICE_SCALE, na * PRICE_SCALE, vb, va])
    return mid, nb, na, vb, va, torch.tensor(x, dtype=torch.float32)


def r(a, n=3):
    return np.round(np.asarray(a, dtype=float), n).ravel().tolist()


class Pair:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = AutoEncoder(DEPTH * 4)
        self.opt = optim.Adam(self.model.parameters(), lr=1e-3)
        self.loss_fn = nn.MSELoss()
        self.hist = deque(maxlen=30)
        self.bids, self.asks = {}, {}

    def reset(self):
        self.bids.clear(); self.asks.clear()

    def apply(self, kind, d):
        if kind == "snapshot":
            self.reset()
        for book, rows in ((self.bids, d["bids"]), (self.asks, d["asks"])):
            for lv in rows:
                p, q = float(lv["price"]), float(lv["qty"])
                if q == 0:
                    book.pop(p, None)
                else:
                    book[p] = q
        for book, pick in ((self.bids, heapq.nlargest), (self.asks, heapq.nsmallest)):
            if len(book) > BOOK_DEPTH:
                keep = set(pick(BOOK_DEPTH, book))
                for p in [p for p in book if p not in keep]:
                    del book[p]

    def tick(self):
        if not (self.bids and self.asks):
            return None
        out = organize_book(self.bids, self.asks)
        if out is None:
            return None
        mid, nb, na, vb, va, row = out
        self.hist.append(mid)
        rate = self.hist[-1] / self.hist[0] - 1.0

        for _ in range(STEPS):
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(row), row)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            m = self.model
            h1 = m.relu(m.layer1(row))
            zt = m.layer2(h1)
            a = m.relu(zt)
            h3 = m.relu(m.layer3(a))
            z = zt.numpy()
        w1, w2, w3, w4 = m.weights()
        # "signal" = each weight times the activation flowing through it right now
        s1 = w1 * row.numpy()[None, :]
        s2 = w2 * h1.numpy()[None, :]
        s3 = w3 * a.numpy()[None, :]
        s4 = w4 * h3.numpy()[None, :]
        pts = np.stack([np.arange(len(z)), z], axis=1)
        eg, ev = np.linalg.eigh(np.cov(pts, rowvar=False))
        order = np.argsort(eg)[::-1]
        eg, ev = eg[order], ev[:, order]
        u = np.linspace(0, 2 * np.pi, 40)
        ell = (np.stack([np.cos(u), np.sin(u)], -1) * 2 * np.sqrt(np.maximum(eg, 0))) @ ev.T + pts.mean(0)

        return {"mid": mid, "rate": float(rate), "loss": float(loss),
                "latent": r(z), "ellipse": r(ell), "nb": r(nb, 6), "na": r(na, 6),
                "vb": r(vb), "va": r(va),
                "w1": r(w1, 4), "w2": r(w2, 4), "w3": r(w3), "w4": r(w4),
                "s1": r(s1), "s2": r(s2), "s3": r(s3), "s4": r(s4)}


pairs = {s: Pair(s) for s in SYMBOLS}
clients: set[WebSocket] = set()


async def kraken_feed():
    sub = {"method": "subscribe",
           "params": {"channel": "book", "symbol": SYMBOLS, "depth": BOOK_DEPTH}}
    while True:
        try:
            async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as s, s.ws_connect(KRAKEN_URL, heartbeat=20) as ws:
                await ws.send_json(sub)
                async for m in ws:
                    if m.type != aiohttp.WSMsgType.TEXT:
                        continue
                    msg = json.loads(m.data)
                    if msg.get("channel") == "book" and msg.get("type") in ("snapshot", "update"):
                        for d in msg["data"]:
                            if d["symbol"] in pairs:
                                pairs[d["symbol"]].apply(msg["type"], d)
        except Exception as e:
            print("Kraken feed error:", e)
        for p in pairs.values():
            p.reset()
        await asyncio.sleep(3)


async def broadcaster():
    while True:
        await asyncio.sleep(TICK)
        if not clients:
            continue
        data = {s: t for s, p in pairs.items() if (t := p.tick())}
        if not data:
            continue
        text = json.dumps({"pairs": data})
        for ws in list(clients):
            try:
                await ws.send_text(text)
            except Exception:
                clients.discard(ws)


@asynccontextmanager
async def lifespan(_):
    tasks = [asyncio.create_task(kraken_feed()), asyncio.create_task(broadcaster())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)
HERE = Path(__file__).parent


@app.get("/")
def index():
    # Works with index.html in ./static/ or sitting next to server.py
    for p in (HERE / "static" / "index.html", HERE / "index.html"):
        if p.exists():
            return FileResponse(p)
    return HTMLResponse("index.html not found next to server.py or in ./static/", status_code=404)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        clients.discard(ws)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)