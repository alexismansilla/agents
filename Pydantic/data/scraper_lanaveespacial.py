"""
Scraper de productos de La Nave Espacial - Categoría Fertilización (Growshop)
URL: https://lanaveespacial.cl/productos/growshop/fertilizacion/

Requiere:
    pip install requests beautifulsoup4

Uso:
    python scraper_lanaveespacial.py

Salidas:
    - fertilizacion.json
    - fertilizacion.csv
"""

import requests
from bs4 import BeautifulSoup
import json
import csv
import re
import time
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# https://lanaveespacial.cl/productos/growshop/fertilizacion/page/2/
# ---------------------- CONFIGURACIÓN ----------------------
BASE_URL = "https://lanaveespacial.cl/productos/growshop/fertilizacion/"
DELAY_ENTRE_PAGINAS = 1.0       # segundos entre requests (sé amable con el servidor)
TIMEOUT = 30                    # segundos
MAX_REINTENTOS = 3
ARCHIVO_JSON = "data/fertilizacion.json"
ARCHIVO_CSV = "data/fertilizacion.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "es-CL,es;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ---------------------- HELPERS ----------------------

def parse_precio(texto):
    """Convierte un texto como '$12.500' o '$1.250.000' a int (CLP)."""
    if not texto:
        return None
    # Buscar todos los grupos de dígitos (con puntos como miles)
    match = re.search(r'[\d\.]+', texto.replace('\xa0', ''))
    if not match:
        return None
    raw = match.group(0).replace('.', '')
    try:
        return int(raw)
    except ValueError:
        return None


def get_page(url, session):
    """GET con reintentos."""
    for intento in range(1, MAX_REINTENTOS + 1):
        try:
            r = session.get(url, timeout=TIMEOUT)
            if r.status_code == 200:
                return r
            if r.status_code == 404:
                return None  # No hay más páginas
            print(f"    ⚠ status {r.status_code} en intento {intento}")
        except requests.RequestException as e:
            print(f"    ⚠ Error de red en intento {intento}: {e}")
        time.sleep(2 * intento)
    return None


def extraer_productos(soup):
    """
    Extrae productos del HTML de una página de categoría WooCommerce.
    Devuelve lista de dicts.
    """
    productos = []

    # WooCommerce: cada producto suele estar en <li class="product ...">
    items = soup.select('li.product')
    if not items:
        # fallback por si cambia la estructura
        items = soup.select('.product-grid-item, .product')

    for item in items:
        # --- Nombre ---
        name_el = (
            item.select_one('.woocommerce-loop-product__title') or
            item.select_one('h2.woocommerce-loop-product__title') or
            item.select_one('h2') or
            item.select_one('h3')
        )
        if not name_el:
            continue
        nombre = name_el.get_text(strip=True)

        # --- Precio actual (si hay descuento, viene dentro de <ins>) ---
        precio_actual_el = (
            item.select_one('.price ins .woocommerce-Price-amount') or
            item.select_one('.price > .woocommerce-Price-amount') or
            item.select_one('.price .woocommerce-Price-amount')
        )
        precio_actual = parse_precio(
            precio_actual_el.get_text() if precio_actual_el else None
        )

        # --- Precio original (si está rebajado) ---
        precio_original_el = item.select_one('.price del .woocommerce-Price-amount')
        precio_original = parse_precio(
            precio_original_el.get_text() if precio_original_el else None
        )

        # --- URL del producto ---
        link_el = (
            item.select_one('a.woocommerce-LoopProduct-link') or
            item.select_one('a.woocommerce-loop-product__link') or
            item.select_one('a')
        )
        url = link_el.get('href') if link_el else None

        # --- ¿Agotado? ---
        agotado = bool(item.select_one('.outofstock, .out-of-stock'))

        producto = {
            "nombre": nombre,
            "precio_clp": precio_actual,
        }
        if precio_original and precio_original != precio_actual:
            producto["precio_original_clp"] = precio_original
            producto["en_oferta"] = True
            if precio_actual:
                descuento_pct = round(
                    (1 - precio_actual / precio_original) * 100, 1
                )
                producto["descuento_pct"] = descuento_pct
        if url:
            producto["url"] = url
        if agotado:
            producto["agotado"] = True

        productos.append(producto)

    return productos


def fetch_details(producto, session):
    """Obtiene la descripción y disponibilidad (stock) desde la página de detalle."""
    url = producto.get("url")
    if not url:
        return producto
    
    r = get_page(url, session)
    if not r:
        return producto
        
    soup = BeautifulSoup(r.text, 'html.parser')
    
    # Descripción
    desc = ""
    desc_div = soup.select_one('.woocommerce-product-details__short-description, .woocommerce-Tabs-panel--description, #tab-description, .elementor-widget-theme-post-excerpt')
    if desc_div:
        desc = desc_div.get_text(separator=' ', strip=True)
        
    if not desc:
        rey_panels = soup.select('.rey-wcPanel-inner')
        for panel in rey_panels:
            text = panel.get_text(separator=' ', strip=True)
            if 'Descripción' in text or len(text) > 20:
                desc = text
                break

    if not desc:
        match = re.search(r'"excerpt"\s*:\s*"([^"]+)"', r.text)
        if match:
            raw_html = match.group(1).encode().decode('unicode_escape')
            desc_soup = BeautifulSoup(raw_html, 'html.parser')
            desc = desc_soup.get_text(separator=' ', strip=True)

    producto["descripcion"] = desc
    
    # Disponibilidad (Stock real desde el max input)
    input_q = soup.find('input', {'name':'quantity'})
    if input_q and input_q.get('max'):
        try:
            producto["disponibilidad"] = int(input_q.get('max'))
        except:
            producto["disponibilidad"] = 1
    elif producto.get("agotado"):
        producto["disponibilidad"] = 0
    else:
        producto["disponibilidad"] = 1 # Si está en stock pero no muestra un max
        
    return producto


def detectar_total_paginas(soup):
    """Lee la paginación de WooCommerce para saber cuántas páginas hay."""
    max_p = 1
    for el in soup.select('.page-numbers'):
        txt = el.get_text(strip=True)
        if txt.isdigit():
            max_p = max(max_p, int(txt))
    return max_p


# ---------------------- MAIN ----------------------

def main():
    session = requests.Session()
    session.headers.update(HEADERS)

    todos_productos = []
    vistos = set()  # para deduplicar por URL

    # Página 1
    print(f"📄 Página 1: {BASE_URL}")
    r = get_page(BASE_URL, session)
    if r is None:
        print("❌ No se pudo obtener la página inicial. Abortando.")
        sys.exit(1)

    soup = BeautifulSoup(r.text, 'html.parser')
    total_paginas = detectar_total_paginas(soup)
    print(f"   Páginas detectadas: {total_paginas}")

    productos = extraer_productos(soup)
    print(f"   Productos encontrados: {len(productos)}")
    for p in productos:
        key = p.get('url') or p['nombre']
        if key not in vistos:
            vistos.add(key)
            todos_productos.append(p)

    # Páginas restantes
    for n in range(2, total_paginas + 1):
        url = f"{BASE_URL}page/{n}/"
        print(f"📄 Página {n}: {url}")
        time.sleep(DELAY_ENTRE_PAGINAS)
        r = get_page(url, session)
        if r is None:
            print(f"   (sin más páginas o error, deteniendo en página {n})")
            break
        soup = BeautifulSoup(r.text, 'html.parser')
        productos = extraer_productos(soup)
        print(f"   Productos encontrados: {len(productos)}")
        if not productos:
            break
        for p in productos:
            key = p.get('url') or p['nombre']
            if key not in vistos:
                vistos.add(key)
                todos_productos.append(p)

    print(f"\nBuscando detalles (descripción y stock) para {len(todos_productos)} productos...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_details, p, session): p for p in todos_productos}
        count = 0
        for future in as_completed(futures):
            future.result()
            count += 1
            print(f"Progreso: {count}/{len(todos_productos)}", end="\r")
    print()

    # Resultado final
    resultado = {
        "fuente": BASE_URL,
        "categoria": "Fertilización - Growshop",
        "moneda": "CLP",
        "total_productos": len(todos_productos),
        "productos": todos_productos,
    }

    # Guardar JSON
    with open(ARCHIVO_JSON, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)

    with open(ARCHIVO_CSV, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "nombre", "precio_clp", "precio_original_clp",
            "en_oferta", "descuento_pct", "agotado", "disponibilidad", "descripcion", "url"
        ])
        for p in todos_productos:
            writer.writerow([
                p.get("nombre", ""),
                p.get("precio_clp", ""),
                p.get("precio_original_clp", ""),
                p.get("en_oferta", ""),
                p.get("descuento_pct", ""),
                p.get("agotado", ""),
                p.get("disponibilidad", ""),
                p.get("descripcion", ""),
                p.get("url", ""),
            ])

    print()
    print("=" * 50)
    print(f"✅ Listo. Total: {len(todos_productos)} productos")
    print(f"   - {ARCHIVO_JSON}")
    print(f"   - {ARCHIVO_CSV}")
    print("=" * 50)

    # Vista previa
    print("\nPrimeros 5 productos:")
    for p in todos_productos[:5]:
        precio = f"${p['precio_clp']:,}".replace(",", ".") if p.get('precio_clp') else "s/precio"
        print(f"  • {p['nombre']} — {precio}")


if __name__ == "__main__":
    main()