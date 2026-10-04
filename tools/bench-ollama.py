#!/usr/bin/env python3
# Benchmark de tok/s de los modelos de Ollama locales.
# - Descarga el modelo activo previo antes de probar cada uno (keep_alive 0).
# - Calienta el modelo con un "hola" sin medirlo.
# - 3 mediciones por configuracion; tok/s = eval_count / eval_duration.
# - Los modelos con capability "thinking" se prueban con y sin razonamiento.
import datetime
import json
import os
import re
import sys
import time
import urllib.request

HOST = "http://localhost:11434"
PROMPT = "Escribe un cuento de unas 150 palabras sobre un gato astronauta."
N = 3
HISTORIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bench-historial.json")

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

def cargar_historial():
    try:
        with open(HISTORIAL, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"mediciones": []}


def guardar_historial(hist):
    with open(HISTORIAL, "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, ensure_ascii=False)
    print(f"\nHistorial guardado en {HISTORIAL}", flush=True)


def ficha_modelo(nombre):
    """Parámetros, familia, cuantización, MoE y capabilities vía /api/show."""
    try:
        show = post("/api/show", {"model": nombre}, timeout=30)
        info = show.get("model_info", {})
        detalles = show.get("details", {})
        expertos_total = expertos_activos = None
        for clave, valor in info.items():
            if clave.endswith(".expert_count") and isinstance(valor, int):
                expertos_total = valor
            elif clave.endswith(".expert_used_count") and isinstance(valor, int):
                expertos_activos = valor
        return {
            "parametros": info.get("general.parameter_count"),
            "familia": detalles.get("family"),
            "cuantizacion": detalles.get("quantization_level"),
            "moe": expertos_total is not None and expertos_total > 1,
            "expertos_total": expertos_total,
            "expertos_activos": expertos_activos,
            "capabilities": show.get("capabilities", []),
        }
    except Exception as e:
        print(f"  aviso: /api/show fallo para {nombre}: {e}", flush=True)
        return {"parametros": None, "familia": None, "cuantizacion": None,
                "moe": None, "expertos_total": None, "expertos_activos": None,
                "capabilities": []}


def ya_medido(hist, modelo, config):
    return any(m.get("modelo") == modelo and m.get("config") == config
               for m in hist.get("mediciones", []))


print("Ollama", get("/api/version").get("version"), flush=True)

forzar = "--forzar" in sys.argv[1:]
if forzar:
    print("Modo --forzar: se repiten todas las mediciones.", flush=True)

models = [m["name"] for m in get("/api/tags")["models"]]
print(f"Modelos: {models}", flush=True)

results = []  # (modelo, config, min, max, media, tokens min-max, truncado?)
hist = cargar_historial()

unload_all()

for m in models:
    print(f"\n=== {m} ===", flush=True)
    unload_all()
    ficha = ficha_modelo(m)
    try:
        caps = post("/api/show", {"name": m}, timeout=30).get("capabilities", [])
    except Exception as e:
        print("  error en /api/show:", e, flush=True)
        continue
    passes = [("sin-think", False), ("con-think", True)] if "thinking" in caps else [("normal", None)]
    if not forzar:
        pendientes = [label for label, _ in passes if not ya_medido(hist, m, label)]
        omitidas = [label for label, _ in passes if ya_medido(hist, m, label)]
        for label in omitidas:
            print(f"  {label}: ya medido (usa --forzar para repetir)", flush=True)
        passes = [(label, think) for label, think in passes if label in pendientes]
        if not passes:
            continue
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
        try:
            tamano = next(x.get("size") for x in get("/api/tags")["models"]
                          if x["name"] == m)
        except Exception:
            tamano = None
        hist["mediciones"].append({
            "fecha": datetime.date.today().isoformat(),
            "modelo": m,
            "config": label,
            "tps_min": round(min(tps_list), 1),
            "tps_max": round(max(tps_list), 1),
            "tps_media": round(sum(tps_list) / len(tps_list), 1),
            "parametros": ficha.get("parametros"),
            "tamano_bytes": tamano,
            "familia": ficha.get("familia"),
            "cuantizacion": ficha.get("cuantizacion"),
            "moe": ficha.get("moe"),
            "expertos_total": ficha.get("expertos_total"),
            "expertos_activos": ficha.get("expertos_activos"),
            "capabilities": ficha.get("capabilities"),
        })

unload_all()
guardar_historial(hist)

# Tabla resumen ordenada por media de tok/s descendente, desde el historial
# (incluye modelos ya medidos en ejecuciones anteriores; los desinstalados
# se marcan para saber que el dato sigue valiendo si se reinstalan).
try:
    instalados_ahora = {m["name"] for m in get("/api/tags").get("models", [])}
except Exception:
    instalados_ahora = set()
resumen = {}
for med in hist.get("mediciones", []):
    clave = (med.get("modelo"), med.get("config"))
    if clave not in resumen or med.get("fecha", "") >= resumen[clave].get("fecha", ""):
        resumen[clave] = med

if resumen:
    print("\n=== RESUMEN (ordenado por tok/s, desde el historial) ===", flush=True)
    print(f"{'Modelo':<22} {'Config':<10} {'Min':>7} {'Max':>7} {'Media':>7}  "
          f"{'Fecha':<10}  Notas", flush=True)
    print("-" * 80, flush=True)
    for med in sorted(resumen.values(), key=lambda r: r.get("tps_media", 0),
                      reverse=True):
        notas = []
        if med.get("moe"):
            notas.append(f"MoE {med.get('expertos_activos')}/"
                         f"{med.get('expertos_total')}")
        if med.get("config") == "con-think":
            notas.append("razonamiento activado")
        if med.get("modelo") not in instalados_ahora:
            notas.append("(desinstalado)")
        print(f"{med.get('modelo', '?'):<22} {med.get('config', '?'):<10} "
              f"{med.get('tps_min', 0):>7.1f} {med.get('tps_max', 0):>7.1f} "
              f"{med.get('tps_media', 0):>7.1f}  "
              f"{med.get('fecha', '?'):<10}  {' '.join(notas)}", flush=True)

print("\nfin", flush=True)
