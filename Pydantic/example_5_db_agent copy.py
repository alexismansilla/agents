import os
import psycopg2
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Any, List
from pydantic import BaseModel
from google import genai

# Cargar variables de entorno desde el archivo .env
load_dotenv()

if 'GEMINI_API_KEY' in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']

# Usamos un dataclass para inyectar dependencias al agente
@dataclass
class DbDependencies:
    conn: Any    # Conexión a psycopg2
    client: Any  # genai.Client para crear embeddings

# Modelos Pydantic para estructurar la salida en JSON
class Producto(BaseModel):
    nombre: str
    precio_clp: int
    stock: int
    url: str
    descripcion_breve: str
    es_el_mas_economico: bool

class RespuestaAgente(BaseModel):
    products: List[Producto]

# Inicializamos el Agente
agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    deps_type=DbDependencies,
    output_type=RespuestaAgente,
    system_prompt=(
        "Eres un asistente experto en productos agrícolas, específicamente en la tienda de La Nave Espacial. "
        "Tu objetivo es ayudar a los usuarios a encontrar los productos que necesitan basándote en la base de datos de la tienda. "
        "SIEMPRE utiliza la herramienta `search_products` para buscar en el catálogo antes de responder o recomendar productos. "
        "Formatea los precios en pesos chilenos (ej: $15.000 CLP) y menciona si hay stock disponible. "
        "REGLA IMPORTANTE: Si el usuario menciona 'el mas barato', 'el mas economico' o términos similares, "
        "debes interpretar que significan lo mismo. En ese caso, debes utilizar la herramienta search_products con limit=5 y find_cheapest=True. "
        "SIEMPRE DEBES DEVOLVER TODOS LOS PRODUCTOS QUE TE ENTREGUE LA HERRAMIENTA. Además, la herramienta te marcará cuál es el más económico, "
        "el cual debe quedar reflejado en la propiedad 'es_el_mas_economico' y debe ser SIEMPRE la primera opción en la lista."
    ),
)

# Definimos una herramienta (tool) que el agente puede usar
@agent.tool
def search_products(ctx: RunContext[DbDependencies], query: str, limit: int = 3, find_cheapest: bool = False) -> str:
    """Busca productos relevantes en la base de datos a partir de una descripción semántica.
    
    Args:
        query: Los términos de búsqueda o descripción de lo que el usuario necesita (ej: "fertilizante para crecimiento").
        limit: La cantidad máxima de productos a retornar (por defecto 3).
        find_cheapest: Si es True, tomará los productos más relevantes y luego los ordenará del más barato al más caro, marcando el de menor precio.
    """
    if find_cheapest:
        limit = 5
        
    print(f"[Tool Executed] Buscando en BD: '{query}' (limit={limit}, find_cheapest={find_cheapest})")
    
    # 1. Convertir la consulta en un vector embedding usando Gemini
    embedding_result = ctx.deps.client.models.embed_content(
        model="gemini-embedding-2",
        contents=[query]
    )
    query_embedding = embedding_result.embeddings[0].values
    
    # 2. Consultar la base de datos (Vector Search)
    cur = ctx.deps.conn.cursor()
    
    # Usamos el operador <=> para ordenamiento por similitud de coseno
    sql = """
    SELECT nombre, precio_clp, descripcion, disponibilidad, url 
    FROM products 
    ORDER BY embedding <=> %s::vector 
    LIMIT %s;
    """
    
    try:
        cur.execute(sql, (query_embedding, limit))
        rows = cur.fetchall()
        
        if not rows:
            return "No se encontraron productos que coincidan con la búsqueda."
            
        if find_cheapest:
            # Ordenar por precio_clp ascendente (índice 1 en la tupla)
            rows = sorted(rows, key=lambda x: x[1])

        # Formatear el resultado para que el Agente lo entienda fácilmente
        resultados_str = "Productos encontrados:\n"
        for idx, row in enumerate(rows, 1):
            nombre, precio, desc, disp, url = row
            extra_tag = " [¡ESTE ES EL MÁS ECONÓMICO!]" if find_cheapest and idx == 1 else ""
            resultados_str += f"{idx}. {nombre}{extra_tag}\n   Precio: ${precio}\n   Stock: {disp}\n   URL: {url}\n   Descripción breve: {desc[:150]}...\n"
            
        return resultados_str
    except Exception as e:
        return f"Error al consultar la base de datos: {str(e)}"
    finally:
        cur.close()

async def main():
    print("Conectando a PostgreSQL...")
    try:
        conn = psycopg2.connect(
            dbname="agents",
            user="myuser",
            password="mypassword",
            host="localhost",
            port="5432"
        )
    except Exception as e:
        print(f"Error conectando a la BD: {e}")
        return

    # Inicializar cliente de Gemini para los embeddings
    client = genai.Client()
    
    # Inyectar las dependencias
    deps = DbDependencies(conn=conn, client=client)
    
    # Hacemos una consulta
    user_query = "Necesito el estimulante de raíces más barato para mis plantas"
    print(f"\nUser: {user_query}")
    print("Agente pensando (y buscando en BD)...\n")
    
    # Ejecutamos el agente
    result = await agent.run(user_query, deps=deps)
    
    print("\n--- Respuesta del Agente ---")
    # Imprimir el resultado en formato JSON
    print(result.output.model_dump_json(indent=2))
    
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
