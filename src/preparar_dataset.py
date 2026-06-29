"""
Script de migración: organiza el dataset PlantVillage en la estructura de clases del proyecto.
Corre UNA sola vez para preparar data/ antes de entrenar.

Fuente: C:\\Users\\nicoa\\Downloads\\archive\\PlantVillage
Destino: data/  (en la raíz del proyecto)
"""

import shutil
import random
from pathlib import Path

ORIGEN = Path(r"C:\Users\nicoa\Downloads\archive\PlantVillage")
DESTINO = Path("data")
CAP_POR_CLASE = 1000

MAPEO = {
    "Tizon_Tardio_Papa":     ["Potato___Late_blight"],
    "Tizon_Temprano_Papa":   ["Potato___Early_blight"],
    "Mancha_Bact_Pimiento":  ["Pepper__bell___Bacterial_spot"],
    "Mancha_Bact_Tomate":    ["Tomato_Bacterial_spot"],
    "Tizon_Temprano_Tomate": ["Tomato_Early_blight"],
    "Tizon_Tardio_Tomate":   ["Tomato_Late_blight"],
    "Moho_Foliar_Tomate":    ["Tomato_Leaf_Mold"],
    "Septoria_Tomate":       ["Tomato_Septoria_leaf_spot"],
    "Arana_Roja_Tomate":     ["Tomato_Spider_mites_Two_spotted_spider_mite"],
    "Mancha_Diana_Tomate":   ["Tomato__Target_Spot"],
    "Virus_Rizo_Tomate":     ["Tomato__Tomato_YellowLeaf__Curl_Virus"],
    "Mosaico_Tomate":        ["Tomato__Tomato_mosaic_virus"],
    "Planta_Sana":           ["Potato___healthy", "Tomato_healthy", "Pepper__bell___healthy"],
}

# Oidio_Vid ya está en data/ desde el dataset original — no se toca

def listar_imagenes(carpeta):
    """Lista imágenes únicas sin duplicar extensiones en Windows."""
    vistas = set()
    resultado = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for img in carpeta.glob(ext):
            key = img.name.lower()
            if key not in vistas:
                vistas.add(key)
                resultado.append(img)
    return resultado

def copiar_clase(nombre_destino, carpetas_origen, cap, forzar=False):
    destino_dir = DESTINO / nombre_destino
    destino_dir.mkdir(parents=True, exist_ok=True)

    ya_existentes = len(list(destino_dir.iterdir()))
    if ya_existentes > 0 and not forzar:
        print(f"  SKIP  {nombre_destino}: ya tiene {ya_existentes} imgs")
        return ya_existentes

    # Limpiar si se fuerza
    if forzar and ya_existentes > 0:
        shutil.rmtree(destino_dir)
        destino_dir.mkdir()

    # Recopilar imágenes de todas las fuentes con prefijo para evitar conflictos
    imagenes = []
    for carpeta in carpetas_origen:
        origen_dir = ORIGEN / carpeta
        if not origen_dir.exists():
            print(f"  WARN  No encontrada: {origen_dir}")
            continue
        for img in listar_imagenes(origen_dir):
            imagenes.append((img, f"{carpeta[:4]}_{img.name}"))  # prefijo corto

    if not imagenes:
        print(f"  ERROR Sin imagenes para {nombre_destino}")
        return 0

    random.seed(42)
    random.shuffle(imagenes)
    seleccionadas = imagenes[:cap]

    for img_path, nuevo_nombre in seleccionadas:
        shutil.copy2(img_path, destino_dir / nuevo_nombre)

    print(f"  OK {nombre_destino}: {len(seleccionadas)} imgs copiadas")
    return len(seleccionadas)

if __name__ == "__main__":
    print(f"Origen: {ORIGEN}")
    print(f"Destino: {DESTINO.resolve()}\n")

    total = 0
    for clase, fuentes in MAPEO.items():
        # Forzar Planta_Sana para reemplazar las 152 existentes
        forzar = (clase == "Planta_Sana")
        n = copiar_clase(clase, fuentes, CAP_POR_CLASE, forzar=forzar)
        total += n

    oidio_dir = DESTINO / "Oidio_Vid"
    oidio_count = len(list(oidio_dir.iterdir())) if oidio_dir.exists() else 0
    print(f"  INFO  Oidio_Vid: {oidio_count} imgs (no se modifica)")
    total += oidio_count

    print(f"\nTotal: {total} imgs en {len(list(DESTINO.iterdir()))} clases")
    print("\nResumen final:")
    for clase_dir in sorted(DESTINO.iterdir()):
        if clase_dir.is_dir():
            count = len(list(clase_dir.iterdir()))
            print(f"  {clase_dir.name:<30} {count}")
