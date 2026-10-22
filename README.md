# LatentTape

![alt](https://raw.githubusercontent.com/MoSolver/LatentTape/main/liverun.gif)

**Autoencoders scanning the crypto order book for hidden patterns, live.**

## What it does

Latent Tape streams real-time order books from Kraken and trains a small **autoencoder** on every USD pair it follows. An autoencoder squeezes each book snapshot down to a handful of numbers and then tries to rebuild the original. To do that well, it has to find the structure that matters in the book: how liquidity is stacked, how lopsided bids and asks are, where the walls sit.

You watch that search happen. The network's layers appear as rotating 3D surfaces, its compressed view of the book appears as a glowing latent-space plot, and the reconstruction loss tells you how well it currently understands the market.

| --- | --- |
| **Live data** | Kraken WebSocket v2, public feed, no API key |
| **Models** | One PyTorch autoencoder per pair, trained online (`40 → 20 → 10 → 20 → 40`) |
| **Pairs** | BTC, ETH, SOL, XRP, ADA, DOGE against USD (easy to change) |
| **Visuals** | three.js 3D weight surfaces, latent ellipse, price tape, order book depth |

> **Note:** this is a visualization and learning project. The model sees one snapshot at a time and has no memory, so it shows what the book looks like right now. It is not a prediction engine and not financial advice.

---

## Quick start

Requires Python 3.9 or newer and a browser with WebGL.

```bash
git clone https://github.com/<your-username>/latent-tape.git
cd latent-tape

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python server.py
```

Open **http://127.0.0.1:8000** and give it a moment to receive the first books.

---

## Using it

- **Click a pair** in the top row to switch the view.
- **Drag a 3D surface** to rotate it. They also turn on their own.
- **Live signal / Raw weights** switches the four layer plots between what is flowing through the network right now and the raw weight values.
- **Green or red latent ellipse** shows whether the price moved up or down over the last few seconds.

---

## Configuration

Edit the constants at the top of `server.py`.

| Constant | Default | What it controls |
| --- | --- | --- |
| `SYMBOLS` | 6 USD pairs | Pairs to stream, in Kraken `BASE/QUOTE` format |
| `DEPTH` | `10` | Book levels per side given to the network |
| `STEPS` | `5` | Training steps per update. Lower it if your CPU struggles. |
| `TICK` | `0.12` | Seconds between updates sent to the browser |

If you change `DEPTH`, also set `const D` in `static/index.html` to the same value.

---

## Project layout

```
latent-tape/
├── server.py          Kraken feed, training, WebSocket broadcast (FastAPI)
├── requirements.txt
├── docs/demo.gif      The showcase recording
└── static/index.html  The whole frontend
```

---

## Troubleshooting

- **"index.html not found":** keep it in `static/` or next to `server.py`.
- **Stuck on "Waiting for the first order books":** the server cannot reach `wss://ws.kraken.com/v2`. Check your network or VPN.
- **3D plots look flat:** three.js loads from a CDN, so you need an internet connection.
- **High CPU:** lower `STEPS` or remove pairs from `SYMBOLS`.

---

Built with [FastAPI](https://fastapi.tiangolo.com/), [PyTorch](https://pytorch.org/) and [three.js](https://threejs.org/). Market data from the [Kraken API](https://docs.kraken.com/api/docs/websocket-v2/book).
