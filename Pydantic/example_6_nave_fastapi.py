import os
import time
import psycopg2.pool
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, FileResponse
from dataclasses import dataclass
from typing import Any, Optional
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


# --- Agent ---

agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    deps_type=DbDependencies,
    system_prompt=(
        "Eres un asistente de ventas para 'La Nave Espacial', una tienda de productos agrícolas. "
        "Tu objetivo es ayudar a los clientes a encontrar productos, consultar stock y calcular presupuestos de forma amable y concisa.\n\n"
        "REGLAS DE NEGOCIO:\n"
        "- Búsquedas específicas: Si el cliente pide lo 'más barato' o 'más económico', debes usar sort_price='asc'. Si pide lo 'más caro', usa sort_price='desc'.\n"
        "- Guardrails (Límites): Si te preguntan por cualquier tema ajeno a agricultura, botánica o la tienda (ej. política, historia, materiales de construcción), "
        "niégate a responder y di EXACTAMENTE: 'Solo puedo ayudarte con productos de La Nave Espacial.'\n"
        "- Formato de productos: Siempre que presentes un producto, MUÉSTRALO EXACTAMENTE con la siguiente estructura (reemplazando los corchetes):\n\n"
        "El producto [más caro/más barato/buscado] es:\n"
        "[Nombre del producto]\n"
        "precio: [Precio] CLP\n"
        "link: [URL]\n"
        "stock: [Stock disponible]\n"
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


# --- App State ---

db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
genai_client: Optional[genai.Client] = None
session_history: dict[str, list] = {}  # historial por session_id


# --- FastAPI App ---

class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool, genai_client
    print("Iniciando recursos...")

    genai_client = genai.Client()

    try:
        db_pool = psycopg2.pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dbname="agents",
            user="myuser",
            password="mypassword",
            host="localhost",
            port="5432",
        )
        print("Pool de conexiones creado.")
    except Exception as e:
        print(f"Error creando pool: {e}")

    yield

    if db_pool:
        db_pool.closeall()
        print("Pool cerrado.")


app = FastAPI(title="La Nave Espacial AI Agent", lifespan=lifespan)


# --- Middleware TTFT ---

@app.middleware("http")
async def ttft_middleware(request: Request, call_next):
    if request.url.path != "/chat/stream":
        return await call_next(request)

    start_time = time.perf_counter()
    response = await call_next(request)

    if isinstance(response, StreamingResponse):
        original_iterator = response.body_iterator

        async def tracked_iterator():
            first_token_sent = False
            async for chunk in original_iterator:
                if not first_token_sent:
                    ttft = time.perf_counter() - start_time
                    print(f"\n[METRICA] TTFT: {ttft:.4f}s")
                    first_token_sent = True
                yield chunk

        response.body_iterator = tracked_iterator()

    return response


# --- Routes ---

@app.get("/")
async def get_chat_html():
    return FileResponse("index.html")


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    conn = db_pool.getconn()

    async def generar_respuesta():
        try:
            deps = DbDependencies(conn=conn, client=genai_client)
            history = session_history.get(request.session_id, [])

            async with agent.run_stream(request.query, deps=deps, message_history=history) as result:
                async for chunk in result.stream_text(delta=True):
                    yield chunk

            session_history[request.session_id] = result.all_messages()
        except Exception as e:
            yield f"\n[Error: {e}]"
        finally:
            db_pool.putconn(conn)

    return StreamingResponse(generar_respuesta(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("example_6_nave_fastapi:app", host="127.0.0.1", port=8000, reload=True)
