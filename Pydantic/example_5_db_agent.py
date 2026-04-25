import os
import psycopg2
import asyncio
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Any
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

# Inicializamos el Agente
agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    deps_type=DbDependencies,
    system_prompt=(
        "Eres un asistente experto en productos agrícolas, específicamente en la tienda de La Nave Espacial. "
        "Tu objetivo es ayudar a los usuarios a encontrar los productos que necesitan basándote en la base de datos de la tienda. "
        "SIEMPRE utiliza la herramienta `search_products` para buscar en el catálogo antes de responder o recomendar productos. "
        "Formatea los precios en pesos chilenos (ej: $15.000 CLP) y menciona si hay stock disponible."
    ),
)

# Definimos una herramienta (tool) que el agente puede usar
@agent.tool
def search_products(ctx: RunContext[DbDependencies], query: str, limit: int = 3) -> str:
    """Busca productos relevantes en la base de datos a partir de una descripción semántica.
    
    Args:
        query: Los términos de búsqueda o descripción de lo que el usuario necesita (ej: "fertilizante para crecimiento").
        limit: La cantidad máxima de productos a retornar (por defecto 3).
    """
    print(f"[Tool Executed] Buscando en BD: '{query}'")
    
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
            
        # Formatear el resultado para que el Agente lo entienda fácilmente
        resultados_str = "Productos encontrados:\n"
        for idx, row in enumerate(rows, 1):
            nombre, precio, desc, disp, url = row
            resultados_str += f"{idx}. {nombre}\n   Precio: ${precio}\n   Stock: {disp}\n   URL: {url}\n   Descripción breve: {desc[:150]}...\n"
            
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
    user_query = "Necesito un estimulante de raíces para mis plantas"
    print(f"\nUser: {user_query}")
    print("Agente pensando (y buscando en BD)...\n")
    
    # Ejecutamos el agente
    result = await agent.run(user_query, deps=deps)
    
    print("\n--- Respuesta del Agente ---")
    print(result.output)
    
    conn.close()

if __name__ == "__main__":
    asyncio.run(main())
