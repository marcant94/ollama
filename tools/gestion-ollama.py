#!/usr/bin/env python3
# Menú interactivo de gestión de modelos de Ollama:
#  - Lista los instalados (/api/tags) y los cargados en memoria (/api/ps).
#  - Arranca un modelo con contexto por defecto, 4K, 8K o 16K.
#  - Descarga modelos de la memoria y borra los instalados.
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = "http://localhost:11434"
HISTORIAL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "bench-historial.json")

CTX_FIJOS = [
    ("4K (4096)", 4096),
    ("8K (8192)", 8192),
    ("16K (16384)", 16384),
]


def limpiar():
    if sys.stdin.isatty() and sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")


def ctx_defecto(nombre):
    """Devuelve (valor, es_nativo) del ctx por defecto del modelo."""
    try:
        show = post("/api/show", {"model": nombre}, timeout=30)
    except Exception:
        return None, False
    import re
    params = show.get("parameters") or ""
    m = re.search(r"(?m)^\s*num_ctx\s+(\d+)", params)
    if m:
        return int(m.group(1)), False
    for clave, valor in (show.get("model_info") or {}).items():
        if clave.endswith(".context_length") and isinstance(valor, int):
            return valor, True
    return None, False


def opciones_ctx(nombre):
    valor, es_nativo = ctx_defecto(nombre)
    if valor is None:
        etiqueta = "defecto del modelo (?)"
    elif es_nativo:
        etiqueta = f"defecto del modelo (nativo {valor})"
    else:
        etiqueta = f"defecto del modelo ({valor})"
    return [(etiqueta, None)] + list(CTX_FIJOS)


def get(path, timeout=30):
    with urllib.request.urlopen(HOST + path, timeout=timeout) as r:
        return json.loads(r.read())


def post(path, payload, timeout=600):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def borrar(path, payload, timeout=120):
    req = urllib.request.Request(HOST + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def gb(n):
    return f"{(n or 0) / 1e9:.2f} GB"


def ultima_medicion(nombre):
    """Media de tok/s más reciente del historial para un modelo (None si no hay)."""
    try:
        with open(HISTORIAL, encoding="utf-8") as f:
            meds = json.load(f).get("mediciones", [])
    except (OSError, ValueError):
        return None
    propias = [m for m in meds if m.get("modelo") == nombre and m.get("tps_media")]
    if not propias:
        return None
    propias.sort(key=lambda m: m.get("fecha", ""), reverse=True)
    return propias[0].get("tps_media")


def instalados():
    return get("/api/tags").get("models", [])


def cargados():
    return {m["name"]: m for m in get("/api/ps").get("models", [])}


def hora_local(iso):
    try:
        base, _, resto = iso.partition(".")
        micro = int(resto.replace("Z", "")[:6].ljust(6, "0") or 0)
        epoch = time.mktime(time.strptime(base, "%Y-%m-%dT%H:%M:%S")) + micro / 1e6
        if time.localtime(epoch).tm_year >= 2100:
            return "sin caducidad"
        return time.strftime("%H:%M", time.localtime(epoch))
    except Exception:
        return "??:??"


def listar():
    modelos = instalados()
    cargas = cargados()
    if not modelos:
        print("No hay modelos instalados.")
        return
    print(f"\n{'#':>3}  {'Modelo':<26} {'Tamaño':>9}  {'tok/s':>7}  Estado")
    print("-" * 80)
    for i, m in enumerate(modelos, 1):
        nombre = m["name"]
        cargado = cargas.get(nombre)
        tps = ultima_medicion(nombre)
        tps_txt = f"{tps:>7.1f}" if tps else f"{'—':>7}"
        if cargado:
            estado = (f"cargado (ctx {cargado.get('context_length', '?')}, "
                      f"RAM {gb(cargado.get('size'))}, "
                      f"VRAM {gb(cargado.get('size_vram'))}, "
                      f"expira {hora_local(cargado.get('expires_at', ''))})")
        else:
            estado = "en disco"
        print(f"{i:>3}  {nombre:<26} {gb(m.get('size')):>9}  {tps_txt}  {estado}")
    print()


def elegir_modelo(modelos, titulo):
    """Muestra una lista numerada y devuelve el nombre elegido (None = cancelar)."""
    if not modelos:
        print("No hay modelos disponibles.")
        return None
    limpiar()
    print(f"\n{titulo}")
    for i, m in enumerate(modelos, 1):
        print(f"  {i}) {m['name']:<26} {gb(m.get('size')):>9}")
    while True:
        eleccion = input("  Modelo (número, 0 = cancelar): ").strip()
        if eleccion in ("0", ""):
            return None
        try:
            idx = int(eleccion) - 1
            if 0 <= idx < len(modelos):
                return modelos[idx]["name"]
        except ValueError:
            pass
        print("  Selección no válida.")


def elegir_opcion(opciones, titulo):
    """Muestra opciones numeradas y devuelve el índice elegido (None = cancelar)."""
    limpiar()
    print(f"\n{titulo}")
    for i, (etiqueta, _) in enumerate(opciones, 1):
        print(f"  {i}) {etiqueta}")
    while True:
        eleccion = input("  Opción (número, 0 = cancelar): ").strip()
        if eleccion in ("0", ""):
            return None
        try:
            idx = int(eleccion) - 1
            if 0 <= idx < len(opciones):
                return idx
        except ValueError:
            pass
        print("  Selección no válida.")


def arrancar():
    nombre = elegir_modelo(instalados(), "Arrancar modelo:")
    if not nombre:
        return
    opciones = opciones_ctx(nombre)
    idx = elegir_opcion(opciones, "Contexto (num_ctx):")
    if idx is None:
        return
    etiqueta, num_ctx = opciones[idx]
    vida_opciones = [("defecto del servidor (10m)", None),
                     ("hasta que lo descargues (-1)", -1)]
    idx_v = elegir_opcion(vida_opciones, "Keep-alive (cuánto permanece en memoria):")
    if idx_v is None:
        return
    keep_alive = vida_opciones[idx_v][1]

    payload = {"model": nombre, "prompt": "", "stream": False}
    options = {"num_predict": 1}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload["options"] = options
    if keep_alive is not None:
        payload["keep_alive"] = keep_alive

    print(f"\nCargando {nombre} (ctx={etiqueta})... puede tardar unos segundos.")
    t0 = time.time()
    try:
        post("/api/generate", payload, timeout=600)
    except urllib.error.HTTPError as e:
        detalle = e.read().decode(errors="replace")
        print(f"Error al cargar {nombre}: HTTP {e.code} {detalle}")
        return
    print(f"{nombre} cargado en {time.time() - t0:.1f}s "
          f"(ctx={etiqueta}, keep_alive={'defecto' if keep_alive is None else keep_alive}).")


def descargar():
    cargas = cargados()
    modelos = list(cargas.values())
    if not modelos:
        print("No hay modelos cargados en memoria.")
        return
    opciones = [(m["name"], m["name"]) for m in modelos] + [("Todos", None)]
    idx = elegir_opcion(opciones, "Descargar de la memoria:")
    if idx is None:
        return
    elegido = opciones[idx][1]
    objetivos = [elegido] if elegido else [m["name"] for m in modelos]
    for nombre in objetivos:
        try:
            post("/api/generate", {"model": nombre, "prompt": "",
                                   "stream": False, "keep_alive": 0}, timeout=60)
            print(f"  {nombre} descargado de la memoria.")
        except Exception as e:
            print(f"  No se pudo descargar {nombre}: {e}")


def eliminar():
    nombre = elegir_modelo(instalados(), "Borrar modelo instalado:")
    if not nombre:
        return
    respuesta = input(f"  ¿Borrar {nombre} y sus pesos del disco? (s/N): ").strip().lower()
    if respuesta not in ("s", "si", "sí", "y"):
        print("Cancelado.")
        return
    try:
        borrar("/api/delete", {"model": nombre, "name": nombre})
        print(f"{nombre} eliminado.")
    except Exception as e:
        print(f"No se pudo borrar {nombre}: {e}")


def main():
    try:
        version = get("/api/version").get("version", "?")
    except Exception:
        print(f"No se pudo conectar a Ollama en {HOST}.")
        print("Arráncalo con: sh dkr-compilar.sh")
        sys.exit(1)

    acciones = {
        "1": ("Listar modelos", listar),
        "2": ("Arrancar modelo (elegir contexto)", arrancar),
        "3": ("Descargar modelo de la memoria", descargar),
        "4": ("Borrar modelo instalado", eliminar),
    }
    while True:
        limpiar()
        print(f"\n=== Ollama {version} — gestión de modelos ===")
        for clave, (etiqueta, _) in acciones.items():
            print(f"  {clave}) {etiqueta}")
        print("  0) Salir")
        try:
            opcion = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if opcion == "0":
            break
        accion = acciones.get(opcion, (None, None))[1]
        if accion is None:
            print("Opción no válida.")
            continue
        try:
            accion()
        except urllib.error.URLError as e:
            print(f"Error de comunicación con Ollama: {e}")
        except KeyboardInterrupt:
            print("\n(interrumpido)")
        try:
            input("\nPulsa Enter para continuar...")
        except (EOFError, KeyboardInterrupt):
            print()
            break


if __name__ == "__main__":
    main()
