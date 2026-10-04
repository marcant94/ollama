#!/usr/bin/env python3
# Estima los tok/s de un modelo SIN descargarlo, calibrado con el historial
# local de benchmarks (tools/bench-historial.json).
#
# Uso:
#   python3 estimar-modelo.py <url de ollama.com o HuggingFace>
#   python3 estimar-modelo.py --params 4B --tamano 2.5GB
#   python3 estimar-modelo.py --params 4B --tamano 2.5GB --ctx 8192
#   python3 estimar-modelo.py --params 8B --tamano 5GB --moe   # fuerza clase MoE
#   python3 estimar-modelo.py --params 3B --tamano 2GB --denso # fuerza clase densa
#
# La estimación es un rango, no un número exacto: la velocidad real depende
# del backend (CPU/Vulkan/CUDA), del offload a VRAM y del contexto elegido.
#
# Modelo físico: en decode cada token obliga a leer todos los pesos, luego
#   tok/s ~= ancho_de_banda_efectivo / tamaño_pesos
# El ancho de banda efectivo se calibra con el historial local (mediana por
# clase denso/MoE). Si el modelo no cabe en VRAM, el cuello de botella pasa
# a ser la RAM (o el PCIe con offload parcial) y la estimación cae en
# consecuencia en lugar de crecer sin límite.
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request

HISTORIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bench-historial.json")
UA = {"User-Agent": "ollama-estimador/1.0"}
BW_RAM = 25e9  # Ryzen 5 3600 DDR4 dual-channel: ~25 GB/s efectivos


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
    mult = {"B": 1e9, "M": 1e6, "K": 1e3}.get(m.group(2) or "B")
    if mult is None:
        raise ValueError(f"parámetros no válidos: {txt!r} (ej: 4B, 0.5B, 8B)")
    return int(num * mult)


def gb(n):
    return f"{n / 1e9:.2f} GB"


def vram_total():
    """VRAM total en bytes (None si no se detecta GPU)."""
    import glob
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_total"):
        try:
            with open(f) as fh:
                total = int(fh.read().strip())
            if total > 0:
                return total
        except (OSError, ValueError):
            continue
    for cmd in (["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits"],
                ["rocm-smi", "--showmeminfo", "vram"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True,
                                 timeout=10).stdout
            m = re.search(r"(\d+)", out)
            if m:
                return int(m.group(1)) * 1024 * 1024
        except (OSError, subprocess.SubprocessError):
            continue
    return None


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


def es_moe(med):
    if med.get("moe") is not None:
        return bool(med.get("moe"))
    fam = (med.get("familia") or "").lower()
    return "moe" in fam


def calibrar(meds):
    """Ancho de banda efectivo (B/s): mediana de tps_media * tamano_bytes."""
    bws = sorted(m["tps_media"] * m["tamano_bytes"] for m in meds)
    return bws[len(bws) // 2]


def estimar(peso, meds, vram):
    """Rango (lo, base, hi) de tok/s con ley BW/peso y penalización por VRAM."""
    bws = sorted(m["tps_media"] * m["tamano_bytes"] for m in meds)
    n = len(bws)
    base_bw = bws[n // 2]
    lo_bw = bws[max(0, n // 2 - n // 4 - 1)]
    hi_bw = bws[min(n - 1, n // 2 + n // 4 + 1)]
    if vram and peso > vram:
        fuera = peso - vram
        bw_ram_ef = min(BW_RAM, base_bw)
        base_bw = 1 / (vram / peso / base_bw + fuera / peso / bw_ram_ef)
        lo_bw = min(lo_bw, base_bw)
        hi_bw = min(hi_bw, base_bw * 1.2)
    return lo_bw / peso, base_bw / peso, hi_bw / peso


def _sin_bloques_tools(template):
    """Quita los bloques condicionales puros sobre tools (Go y Jinja, con else).

    Solo recorta el `if` cuya condición es exactamente `.Tools`/`tools`
    (no `if or .System .Tools`, que se activa también sin tools): elimina
    la rama `if` y conserva la rama `else`/`elif`, que es lo que se
    renderiza sin herramientas. Empareja por nivel de anidamiento.
    """
    estilos = [
        {"abre": "{{", "cierra": "}}", "cond": ".Tools",
         "fines": ("end",), "aperturas": ("if", "range")},
        {"abre": "{%", "cierra": "%}", "cond": "tools",
         "fines": ("endif", "endfor"), "aperturas": ("if", "for")},
    ]
    for est in estilos:
        tag = re.compile(re.escape(est["abre"]) + r"(.*?)"
                         + re.escape(est["cierra"]), re.S)
        tokens = [(m.start(), m.end(), m.group(1).strip(" -").split())
                  for m in tag.finditer(template)]
        salida, i, nivel = [], 0, None
        for ini_pos, fin_pos, palabras in tokens:
            if not palabras:
                continue
            clave = palabras[0].lower()
            resto = " ".join(palabras[1:])
            es_ini_tools = (clave == "if" and
                            resto.strip().lower() == est["cond"].lower())
            if nivel is None and es_ini_tools:
                salida.append(template[i:ini_pos])
                nivel, i, buscando_else = 1, None, True
                continue
            if nivel is not None:
                if clave in est["aperturas"]:
                    nivel += 1
                elif clave in ("else", "elif") or (
                        clave == "else" and resto.lower().startswith("if")):
                    if nivel == 1 and buscando_else:
                        i, buscando_else = fin_pos, False
                elif clave in est["fines"]:
                    nivel -= 1
                    if nivel == 0:
                        nivel = None
                        if i is not None:
                            salida.append(template[i:ini_pos])
                        i = fin_pos
                elif clave == "else" and nivel == 1 and buscando_else:
                    i, buscando_else = fin_pos, False
        if nivel is None:
            template = "".join(salida) + template[i:]
    return template


def _cond_compuesta_tools(template):
    """True si algún `if` puede cumplirse mencionando Tools SIN herramientas.

    Un `if or .System .Tools` (estilo hermes3) se activa aunque la
    petición no traiga herramientas: el template condiciona la charla al
    andamiaje de tools. En cambio una conjunción (`if .Tools and ...`,
    `if tools and tools is iterable ...`) exige tools, igual que el
    `if .Tools` exacto (que aparte se elimina con _sin_bloques_tools).
    Heurístico a propósito: mejor excluir una medición dudosa que
    contaminar la calibración.
    """
    estilos = [("{%", "%}", r"\btools\b"), ("{{", "}}", r"\.Tools\b")]
    for abre, cierra, cond in estilos:
        tag = re.compile(re.escape(abre) + r"(.*?)" + re.escape(cierra), re.S)
        for m in tag.finditer(template):
            palabras = m.group(1).strip(" -").split()
            if not palabras or palabras[0].lower() != "if":
                continue
            resto = " ".join(palabras[1:]).strip()
            if not re.search(cond, resto, re.I):
                continue
            simple = re.sub(r"[(){}$]", "", resto).strip()
            if simple.lower() in (".tools", "tools"):
                continue
            # Go: `or` de prefijo (variadic). Si alguna rama no menciona
            # tools, la condición se cumple también sin herramientas.
            if len(palabras) > 1 and re.match(r"\(*or\b", palabras[1], re.I):
                ramas = [p.lower() for p in palabras[2:]]
                if any(not re.search(cond, r, re.I) for r in ramas):
                    return True
                continue
            # Jinja/Go infix: `a or b`.
            ramas = re.split(r"\s+or\s+", resto, flags=re.I)
            if len(ramas) > 1 and any(
                    not re.search(cond, r, re.I) for r in ramas):
                return True
    return False


def es_conversacional(med):
    """Detecta si el modelo charla normal sin tools (dinámico, por template).

    Señales: (1) el template menciona Tools en una condición compuesta,
    que se cumple sin herramientas (p. ej. hermes3), o (2) tras quitar los
    bloques `if .Tools` quedan instrucciones de function-calling. En ambos
    casos la medición no es charla comparable. Sin template: se asume
    conversacional.
    """
    template = med.get("template") or ""
    if not template:
        return True
    if _cond_compuesta_tools(template):
        return False
    sin_cond = _sin_bloques_tools(template)
    restos = re.findall(r"function calling|"
                        r"You may call one or more functions|"
                        r"tool call you will make",
                        sin_cond, flags=re.I)
    return not restos


def grupo_relevante(meds, es_objetivo_moe):
    """Mediciones conversacionales de la misma clase (MoE/denso) que el objetivo."""
    conv = [m for m in meds if es_conversacional(m)]
    grupo = [m for m in conv if es_moe(m) == es_objetivo_moe]
    if len(grupo) >= 2:
        return grupo, []
    return (conv if len(conv) >= 2 else meds), [
        m for m in meds if not es_conversacional(m)]


def main(argv):
    params = tamano = ctx = None
    objetivo = None
    es_objetivo_moe = None
    args = iter(argv[1:])
    for a in args:
        if a == "--params":
            params = parse_params(next(args))
        elif a == "--tamano":
            tamano = parse_tamano(next(args))
        elif a == "--ctx":
            ctx = int(next(args))
        elif a == "--moe":
            es_objetivo_moe = True
        elif a == "--denso":
            es_objetivo_moe = False
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

    if es_objetivo_moe is None:
        es_objetivo_moe = bool(re.search(r"moe|mixtral|deepseek|qwen3[.-]?\d.*a\d|bailing",
                                         etiqueta, re.I))
    grupo, excluidos = grupo_relevante(meds, es_objetivo_moe)
    clase = "MoE" if es_objetivo_moe else "denso"
    if excluidos:
        print(f"Excluidos del calibrado (no conversacionales): "
              f"{', '.join(sorted({m['modelo'] for m in excluidos}))}.")
    if len(grupo) < len(meds):
        print(f"Calibrando con {len(grupo)} mediciones de modelos "
              f"{clase}s (de {len(meds)} totales).")
    else:
        print(f"Historial: {len(meds)} mediciones "
              f"({', '.join(sorted({m['modelo'] for m in meds}))})")
        print("Aviso: historial sin variedad MoE/denso suficiente; "
              "la estimación mezcla ambas clases.")

    print(f"Parámetros: {params / 1e9:.1f}B" if params else "Parámetros: ?",
          f"· Tamaño: {gb(tamano)}",
          f"· ctx: {ctx}" if ctx else "")
    vram = vram_total()
    if vram:
        print(f"VRAM detectada: {gb(vram)}", end="")
        if tamano > vram:
            print(f" — el modelo ({gb(tamano)}) NO cabe entero: "
                  f"{gb(tamano - vram)} caerían a RAM (lento).")
        else:
            print(" — el modelo cabe entero en VRAM.")
    else:
        print("VRAM: no detectada (se asume que el modelo cabe en VRAM).")
    print(f"Ancho de banda efectivo calibrado ({clase}): "
          f"{calibrar(grupo) / 1e9:.0f} GB/s (mediana del grupo)")

    lo, base, hi = estimar(tamano, grupo, vram)
    print(f"\nEstimación para {etiqueta}: {lo:.0f}-{hi:.0f} tok/s "
          f"(central ~{base:.0f} tok/s)")
    if ctx and ctx > 8192:
        print("Aviso: con ctx > 8K el KV-cache crece y parte del modelo puede "
              "caer a RAM: la velocidad real puede quedar por debajo del rango.")
    print("Rango orientativo: la velocidad real depende del backend, del "
          "offload a VRAM y del contexto elegido.")


if __name__ == "__main__":
    main(sys.argv)
