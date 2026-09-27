#!/usr/bin/env python3
# Benchmark de tok/s de los modelos de Ollama locales.
# - Descarga el modelo activo previo antes de probar cada uno (keep_alive 0).
# - Calienta el modelo con un "hola" sin medirlo.
# - 3 mediciones por configuracion; tok/s = eval_count / eval_duration.
# - Los modelos con capability "thinking" se prueban con y sin razonamiento.
import json, time, urllib.request

HOST = "http://localhost:11434"
PROMPT = "Escribe un cuento de unas 150 palabras sobre un gato astronauta."
N = 3

def post(path, payload, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())

def get(path):
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.loads(r.read())

def loaded_models():
    try:
        return [m["name"] for m in get("/api/ps").get("models", [])]
    except Exception as e:
        print("aviso: /api/ps fallo:", e, flush=True)
        return []

def unload(name):
    try:
        post("/api/generate", {"model": name, "keep_alive": 0}, timeout=60)
        print(f"  (descargado {name})", flush=True)
    except Exception as e:
        print(f"  aviso: no se pudo descargar {name}: {e}", flush=True)

def unload_all():
    for m in loaded_models():
        unload(m)

print("Ollama", get("/api/version").get("version"), flush=True)

models = [m["name"] for m in get("/api/tags")["models"]]
print(f"Modelos: {models}", flush=True)

results = []  # (modelo, config, min, max, media, tokens min-max, truncado?)

unload_all()

for m in models:
    print(f"\n=== {m} ===", flush=True)
    unload_all()
    try:
        caps = post("/api/show", {"name": m}, timeout=30).get("capabilities", [])
    except Exception as e:
        print("  error en /api/show:", e, flush=True)
        continue
    passes = [("sin-think", False), ("con-think", True)] if "thinking" in caps else [("normal", None)]
    for label, think in passes:
        # Warm-up (no se mide): carga el modelo en VRAM.
        wu = {"model": m, "messages": [{"role": "user", "content": "hola"}], "stream": False}
        if think is not None:
            wu["think"] = think
        try:
            post("/api/chat", wu)
        except Exception as e:
            print(f"  {label}: warmup fallo: {e}", flush=True)
            break
        tps_list, ec_list = [], []
        for k in range(N):
            payload = {"model": m, "stream": False,
                       "messages": [{"role": "user", "content": PROMPT}],
                       "options": {"temperature": 0.6, "num_predict": 1024}}
            if think is not None:
                payload["think"] = think
            t0 = time.time()
            r = post("/api/chat", payload)
            wall = time.time() - t0
            ed, ec = r.get("eval_duration", 0), r.get("eval_count", 0)
            tps = ec / (ed / 1e9) if ed else 0
            reasoning = r.get("message", {}).get("thinking", "") or ""
            tps_list.append(tps); ec_list.append(ec)
            print(f"  {label} #{k+1}: {tps:.1f} tok/s, {ec} tokens, {wall:.1f}s, "
                  f"razonamiento={len(reasoning)} chars", flush=True)
        print(f"  >> {label}: min={min(tps_list):.1f} max={max(tps_list):.1f} "
              f"media={sum(tps_list)/len(tps_list):.1f} tok/s "
              f"(tokens gen {min(ec_list)}-{max(ec_list)})", flush=True)
        trunc = max(ec_list) >= 1024
        results.append((m, label, min(tps_list), max(tps_list),
                        sum(tps_list) / len(tps_list), min(ec_list), max(ec_list), trunc))

unload_all()

# Tabla resumen ordenada por media de tok/s descendente.
if results:
    print("\n=== RESUMEN (ordenado por tok/s) ===", flush=True)
    print(f"{'Modelo':<22} {'Config':<10} {'Min':>7} {'Max':>7} {'Media':>7}  "
          f"{'Tokens':>9}  Notas", flush=True)
    print("-" * 80, flush=True)
    for m, label, mn, mx, avg, ecmin, ecmax, trunc in \
            sorted(results, key=lambda r: r[4], reverse=True):
        notas = []
        if trunc:
            notas.append("truncado (num_predict)")
        if any(c in label for c in "think") and label == "con-think":
            pass
        if label == "con-think":
            notas.append("razonamiento activado")
        elif label == "sin-think":
            notas.append("")
        elif label == "normal":
            notas.append("")
        notas_str = " ".join(n for n in notas if n)
        print(f"{m:<22} {label:<10} {mn:>7.1f} {mx:>7.1f} {avg:>7.1f}  "
              f"{ecmin}-{ecmax:>6}  {notas_str}", flush=True)

print("\nfin", flush=True)
