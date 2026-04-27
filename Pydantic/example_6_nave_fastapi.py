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

MAX_HISTORY_MESSAGES = 6
MAX_EMBEDDING_CACHE = 200

_embedding_cache: dict[str, list] = {}

# dev  → groq:llama-3.3-70b-versatile  (14,400 req/día gratis, muy rápido)
# prod → google-gla:gemini-2.5-flash-lite
_ENV = os.getenv("APP_ENV", "dev")
_MODEL = "groq:llama-3.3-70b-versatile" if _ENV == "dev" else "google-gla:gemini-2.5-flash-lite"
print(f"[CONFIG] entorno={_ENV} modelo={_MODEL}")


# --- Dependencies ---

@dataclass
class DbDependencies:
    conn: Any
    client: Any


# --- Agent ---

agent = Agent(
    model=_MODEL,
    deps_type=DbDependencies,
    system_prompt=(
        "Eres un asistente de ventas de 'La Nave Espacial', tienda de productos agrícolas. "
        "Usa SIEMPRE las herramientas disponibles para buscar productos antes de responder. "
        "Si piden el más barato usa sort_price='asc', el más caro usa sort_price='desc'. "
        "Al mostrar un producto incluye: nombre, precio en CLP, link y stock. "
        "Si te preguntan algo ajeno a agricultura o la tienda, responde solo: "
        "'Solo puedo ayudarte con productos de La Nave Espacial.'"
    ),
)


# --- Helpers ---

def get_embedding(client: Any, query: str) -> list:
    if query not in _embedding_cache:
        if len(_embedding_cache) >= MAX_EMBEDDING_CACHE:
            _embedding_cache.clear()
        _embedding_cache[query] = client.models.embed_content(
            model="gemini-embedding-2",
            contents=[query],
        ).embeddings[0].values
    return _embedding_cache[query]


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

    print(f"[search_products] query='{query}' limit={limit} sort_price={sort_price} cache_size={len(_embedding_cache)}")

    embedding = get_embedding(ctx.deps.client, query)

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

        lines = ["Productos:"]
        for idx, (nombre, precio, desc, stock, url) in enumerate(rows, 1):
            tag = f" {tags[sort_price]}" if sort_price != "none" and idx == 1 else ""
            lines.append(
                f"{idx}.{tag} {nombre} | ${precio:,} CLP | stock:{stock} | {url} | {desc[:80]}"
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
            history = session_history.get(request.session_id, [])[-MAX_HISTORY_MESSAGES:]

            async with agent.run_stream(request.query, deps=deps, message_history=history) as result:
                async for chunk in result.stream_text(delta=True):
                    yield chunk

            all_msgs = result.all_messages()
            session_history[request.session_id] = all_msgs
            usage = result.usage()
            print(f"[TOKENS] request={usage.request_tokens} response={usage.response_tokens} total={usage.total_tokens}")
        except Exception as e:
            yield f"\n[Error: {e}]"
        finally:
            db_pool.putconn(conn)

    return StreamingResponse(generar_respuesta(), media_type="text/event-stream")


@app.delete("/chat/history/{session_id}")
async def clear_history(session_id: str):
    session_history.pop(session_id, None)
    return {"cleared": session_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("example_6_nave_fastapi:app", host="127.0.0.1", port=8000, reload=True)
