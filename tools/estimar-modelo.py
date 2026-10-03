#!/usr/bin/env python3
# Estima los tok/s de un modelo SIN descargarlo, calibrado con el historial
# local de benchmarks (tools/bench-historial.json).
#
# Uso:
#   python3 estimar-modelo.py <url de ollama.com o HuggingFace>
#   python3 estimar-modelo.py --params 4B --tamano 2.5GB
#   python3 estimar-modelo.py --params 4B --tamano 2.5GB --ctx 8192
#
# La estimación es un rango, no un número exacto: la velocidad real depende
# del backend (CPU/Vulkan/CUDA), del offload a VRAM y del contexto elegido.
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HISTORIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bench-historial.json")
UA = {"User-Agent": "ollama-estimador/1.0"}


def http_json(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def parse_tamano(txt):
    m = re.fullmatch(r"\s*([\d.,]+)\s*([KMGT]?B)\s*", txt.upper())
    if not m:
        raise ValueError(f"tamaño no válido: {txt!r} (ej: 2.5GB, 800MB)")
    num = float(m.group(1).replace(",", "."))
    return int(num * {"B": 1, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}[m.group(2)])


def parse_params(txt):
    m = re.fullmatch(r"\s*([\d.,]+)\s*([BMK])?\s*", txt.upper())
    if not m:
        raise ValueError(f"parámetros no válidos: {txt!r} (ej: 4B, 0.5B, 8B)")
    num = float(m.group(1).replace(",", "."))
    return int(num * {"B": 1, "M": 1e6, "K": 1e3}.get(m.group(2) or "B", 1))


def gb(n):
    return f"{n / 1e9:.2f} GB"


def resolver_ollama(url):
    """URL tipo https://ollama.com/library/qwen3:4b -> (params, tamaño, etiqueta)."""
    ruta = urllib.parse.urlparse(url).path.strip("/")
    partes = ruta.split("/")
    if len(partes) < 2 or partes[0] != "library":
        raise ValueError("URL de ollama.com no reconocida "
                         "(esperaba https://ollama.com/library/<modelo>[:tag])")
    repo, _, tag = partes[1].partition(":")
    tag = tag or "latest"
    manifest = http_json(f"https://registry.ollama.ai/v2/library/{repo}/manifests/{tag}")
    capas = manifest.get("layers", [])
    peso = next((c["size"] for c in capas
                 if c.get("mediaType") == "application/vnd.ollama.image.model"), None)
    if peso is None:
        raise ValueError(f"el manifiesto de {repo}:{tag} no trae capa de modelo")
    digest = manifest["config"]["digest"]
    cfg = http_json(f"https://registry.ollama.ai/v2/library/{repo}/blobs/{digest}")
    tipo = cfg.get("model_type", "")
    m = re.fullmatch(r"\s*([\d.,]+)\s*([BMK])\s*", tipo.upper())
    params = parse_params(m.group(1) + m.group(2)) if m else None
    return params, peso, f"{repo}:{tag} ({cfg.get('file_type', '?')})"


def resolver_hf(url):
    """URL tipo https://huggingface.co/<org>/<repo>[/blob/.../<fichero>.gguf]."""
    ruta = urllib.parse.urlparse(url).path.strip("/").split("/")
    if len(ruta) < 2:
        raise ValueError("URL de HuggingFace no reconocida")
    repo_id = f"{ruta[0]}/{ruta[1]}"
    meta = http_json(f"https://huggingface.co/api/models/{repo_id}?blobs=false")
    gguf_total = (meta.get("gguf") or {}).get("total")
    fichero = None
    if "blob" in ruta or "resolve" in ruta:
        fichero = ruta[-1]
    if fichero and fichero.endswith(".gguf"):
        blobs = http_json(f"https://huggingface.co/api/models/{repo_id}?blobs=true")
        hermanos = {s.get("rfilename", "").split("/")[-1]: s.get("size")
                     for s in blobs.get("siblings", [])}
        peso = hermanos.get(fichero)
        if peso is None:
            raise ValueError(f"{fichero} no aparece en {repo_id}")
        etiqueta = f"{repo_id}/{fichero}"
    else:
        hermanos = http_json(
            f"https://huggingface.co/api/models/{repo_id}?blobs=true").get("siblings", [])
        ggufs = [(s.get("rfilename", ""), s.get("size")) for s in hermanos
                 if s.get("rfilename", "").endswith(".gguf") and s.get("size")]
        if not ggufs:
            raise ValueError(f"{repo_id} no publica ningún .gguf")
        print("Ficheros GGUF disponibles:")
        for i, (nombre, tam) in enumerate(sorted(ggufs), 1):
            print(f"  {i}) {nombre.split('/')[-1]}  ({gb(tam)})")
        eleccion = input("Elige el fichero (número): ").strip()
        nombre, peso = sorted(ggufs)[int(eleccion) - 1]
        etiqueta = f"{repo_id}/{nombre}"
    return gguf_total, peso, etiqueta


def cargar_historial():
    try:
        with open(HISTORIAL, encoding="utf-8") as f:
            meds = json.load(f).get("mediciones", [])
    except (OSError, ValueError):
        return []
    return [m for m in meds if m.get("tps_media") and m.get("tamano_bytes")]


def calibrar(meds):
    """Eficiencia observada: tok/s reales por GB de pesos (mediana)."""
    ratios = sorted(m["tps_media"] / (m["tamano_bytes"] / 1e9) for m in meds)
    return ratios[len(ratios) // 2]


def estimar(peso, meds):
    efi = calibrar(meds)
    base = efi * (peso / 1e9)
    ratios = sorted(m["tps_media"] / (m["tamano_bytes"] / 1e9) for m in meds)
    p25, p75 = ratios[len(ratios) // 4], ratios[3 * len(ratios) // 4]
    return base, p25 * (peso / 1e9), p75 * (peso / 1e9)


def main(argv):
    params = tamano = ctx = None
    objetivo = None
    args = iter(argv[1:])
    for a in args:
        if a == "--params":
            params = parse_params(next(args))
        elif a == "--tamano":
            tamano = parse_tamano(next(args))
        elif a == "--ctx":
            ctx = int(next(args))
        elif a.startswith("-"):
            sys.exit(f"Opción desconocida: {a}")
        elif objetivo is None:
            objetivo = a
        else:
            sys.exit("Demasiados argumentos posicionales.")

    if objetivo:
        host = urllib.parse.urlparse(objetivo).netloc
        try:
            if "ollama.com" in host:
                params, tamano, etiqueta = resolver_ollama(objetivo)
            elif "huggingface.co" in host:
                params, tamano, etiqueta = resolver_hf(objetivo)
            else:
                sys.exit("URL no soportada (solo ollama.com y huggingface.co).")
        except (ValueError, IndexError) as e:
            sys.exit(f"Error: {e}")
        print(f"Modelo: {etiqueta}")
    else:
        if tamano is None:
            sys.exit("Indica una URL o al menos --tamano (ej: --tamano 2.5GB).")
        etiqueta = "modelo manual"

    meds = cargar_historial()
    if not meds:
        sys.exit(f"No hay historial en {HISTORIAL}.\n"
                 "Ejecuta primero: python3 tools/bench-ollama.py")

    print(f"Parámetros: {params / 1e9:.1f}B" if params else "Parámetros: ?",
          f"· Tamaño: {gb(tamano)}",
          f"· ctx: {ctx}" if ctx else "")
    print(f"Historial: {len(meds)} mediciones "
          f"({', '.join(sorted({m['modelo'] for m in meds}))})")
    print(f"Eficiencia calibrada: {calibrar(meds):.1f} tok/s por GB de pesos "
          "(mediana del historial)")

    base, lo, hi = estimar(tamano, meds)
    print(f"\nEstimación para {etiqueta}: {lo:.0f}-{hi:.0f} tok/s "
          f"(central ~{base:.0f} tok/s)")
    if ctx and ctx > 8192:
        print("Aviso: con ctx > 8K el KV-cache crece y parte del modelo puede "
              "caer a RAM: la velocidad real puede quedar por debajo del rango.")
    print("Rango orientativo: la velocidad real depende del backend, del "
          "offload a VRAM y del contexto elegido.")


if __name__ == "__main__":
    main(sys.argv)
