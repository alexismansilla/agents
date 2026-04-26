import os
import asyncio
import psycopg2
from dataclasses import dataclass
from typing import Any, List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext
from google import genai

load_dotenv()

if "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# --- Dependencies ---

@dataclass
class DbDependencies:
    conn: Any
    client: Any


# --- Output Models ---

class Producto(BaseModel):
    nombre: str
    precio_clp: int
    stock: int
    url: str
    descripcion_breve: str
    destacado: bool = False  # True si es el más barato o más caro según el contexto

class RespuestaAgente(BaseModel):
    productos: Optional[List[Producto]] = None
    mensaje: Optional[str] = None  # Para respuestas puntuales como stock o fuera de scope


# --- Agent ---

agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    deps_type=DbDependencies,
    output_type=RespuestaAgente,
    system_prompt=(
        "Eres un asistente de ventas para 'La Nave Espacial', una tienda de productos agrícolas. "
        "Tu objetivo es ayudar a los clientes a encontrar productos, consultar stock y calcular presupuestos de forma amable y concisa.\n\n"
        "REGLAS DE NEGOCIO:\n"
        "- Búsquedas específicas: Si el cliente pide lo 'más barato' o 'más económico', debes usar sort_price='asc'. Si pide lo 'más caro', usa sort_price='desc'.\n"
        "- Guardrails (Límites): Si te preguntan por cualquier tema ajeno a agricultura, botánica o la tienda (ej. política, historia, materiales de construcción), "
        "niégate a responder y di EXACTAMENTE: 'Solo puedo ayudarte con productos de La Nave Espacial.'"
    ),
)


# --- Tools ---

@agent.tool
def search_products(
    ctx: RunContext[DbDependencies],
    query: str,
    limit: int = 3,
    sort_price: str = "none",
) -> str:
    """Busca productos por similitud semántica.

    Args:
        query: Descripción de lo que busca el usuario.
        limit: Máximo de productos a retornar.
        sort_price: "asc" (más barato), "desc" (más caro), "none" (por relevancia).
    """
    if sort_price in ("asc", "desc"):
        limit = 5

    print(f"[search_products] query='{query}' limit={limit} sort_price={sort_price}")

    embedding = ctx.deps.client.models.embed_content(
        model="gemini-embedding-2",
        contents=[query],
    ).embeddings[0].values

    cur = ctx.deps.conn.cursor()
    try:
        cur.execute(
            """
            SELECT nombre, precio_clp, descripcion, disponibilidad, url
            FROM products
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (embedding, limit),
        )
        rows = cur.fetchall()

        if not rows:
            return "No se encontraron productos."

        if sort_price == "asc":
            rows = sorted(rows, key=lambda r: r[1])
        elif sort_price == "desc":
            rows = sorted(rows, key=lambda r: r[1], reverse=True)

        tags = {"asc": "[MÁS ECONÓMICO]", "desc": "[MÁS CARO]", "none": ""}

        lines = ["Productos encontrados:"]
        for idx, (nombre, precio, desc, stock, url) in enumerate(rows, 1):
            tag = f" {tags[sort_price]}" if sort_price != "none" and idx == 1 else ""
            lines.append(
                f"{idx}. {nombre}{tag}\n"
                f"   Precio: ${precio:,} CLP | Stock: {stock}\n"
                f"   URL: {url}\n"
                f"   Descripción: {desc[:150]}..."
            )

        return "\n".join(lines)
    except Exception as e:
        return f"Error en búsqueda: {e}"
    finally:
        cur.close()


@agent.tool
def get_stock(ctx: RunContext[DbDependencies], product_name: str) -> str:
    """Retorna el stock exacto de un producto específico buscando por nombre.

    Args:
        product_name: Nombre o parte del nombre del producto.
    """
    print(f"[get_stock] product_name='{product_name}'")

    cur = ctx.deps.conn.cursor()
    try:
        cur.execute(
            "SELECT nombre, disponibilidad FROM products WHERE nombre ILIKE %s LIMIT 1",
            (f"%{product_name}%",),
        )
        row = cur.fetchone()

        if not row:
            return f"No se encontró el producto '{product_name}'."

        nombre, stock = row
        return f"Producto: {nombre} | Stock: {stock} unidades"
    except Exception as e:
        return f"Error consultando stock: {e}"
    finally:
        cur.close()


@agent.tool
def calculate_cost(ctx: RunContext[DbDependencies], price: float, quantity: int, discount_percent: float = 0.0) -> str:
    """Calcula el costo total dados un precio unitario, cantidad y porcentaje de descuento.

    Args:
        price: Precio unitario del producto.
        quantity: Cantidad de unidades a comprar.
        discount_percent: Porcentaje de descuento a aplicar (ej. 10 para 10%).
    """
    print(f"[calculate_cost] price={price} quantity={quantity} discount_percent={discount_percent}")
    total = price * quantity
    discount_amount = total * (discount_percent / 100)
    final_total = total - discount_amount
    return f"Total sin descuento: ${total:,.2f}. Descuento: ${discount_amount:,.2f}. Total final: ${final_total:,.2f} CLP."


# --- Main ---

QUERIES = [
    "Stock de PLAGRON – Power Roots (250 ml)",
    "Necesito el fertilizante más barato",
    "Cuál es el producto más caro",
    "Fertilizante para la etapa de crecimiento",
    "Alternativas al PLAGRON Power Roots",
    "¿Cuánto nos costaría comprar 5 PLAGRON Power Roots (250 ml) si tenemos 10% de descuento?",
    "¿Cuánto nos costaría comprar 5 vigas si tenemos 10% de descuento?",
    "¿Cuál es la capital de Francia?",
]


async def run_query(query: str, deps: DbDependencies) -> None:
    print(f"\n{'='*60}")
    print(f"User: {query}")
    result = await agent.run(query, deps=deps)
    print(result.output.model_dump_json(indent=2))


async def main():
    print("Conectando a PostgreSQL...")
    try:
        conn = psycopg2.connect(
            dbname="agents",
            user="myuser",
            password="mypassword",
            host="localhost",
            port="5432",
        )
    except Exception as e:
        print(f"Error conectando a la BD: {e}")
        return

    deps = DbDependencies(conn=conn, client=genai.Client())

    try:
        for query in QUERIES:
            await run_query(query, deps)
    finally:
        conn.close()


if __name__ == "__main__":
    asyncio.run(main())
