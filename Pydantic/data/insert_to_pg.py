import json
import psycopg2
import sys
import os
from google import genai
from pgvector.psycopg2 import register_vector

def load_env():
    # Cargar GEMINI_API_KEY desde .env de forma manual
    try:
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("GEMINI_API_KEY"):
                    key = line.strip().split("=", 1)[1]
                    os.environ["GEMINI_API_KEY"] = key
    except Exception as e:
        print("No se pudo cargar el archivo .env:", e)

def main():
    load_env()
    if not os.environ.get("GEMINI_API_KEY"):
        print("Error: No se encontró GEMINI_API_KEY en las variables de entorno.")
        sys.exit(1)

    json_path = "data/fertilizacion.json"
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {json_path}")
        sys.exit(1)

    productos = data.get("productos", [])
    if not productos:
        print("No hay productos para procesar.")
        sys.exit(0)

    print("Conectando a PostgreSQL...")
    try:
        conn = psycopg2.connect(
            dbname="agents",
            user="myuser",
            password="mypassword",
            host="localhost",
            port="5432"
        )
    except psycopg2.Error as e:
        print("Error conectando a la base de datos PostgreSQL:", e)
        sys.exit(1)

    cur = conn.cursor()
    
    # Habilitar extensión y registrar tipo vector
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    register_vector(conn)

    # Crear tabla
    print("Creando tabla 'products'...")
    cur.execute("DROP TABLE IF EXISTS products;")
    create_table_query = """
    CREATE TABLE IF NOT EXISTS products (
        id SERIAL PRIMARY KEY,
        nombre VARCHAR(500),
        precio_clp INTEGER,
        url VARCHAR(1000),
        descripcion TEXT,
        disponibilidad INTEGER,
        precio_original_clp INTEGER,
        en_oferta BOOLEAN,
        descuento_pct NUMERIC,
        agotado BOOLEAN,
        embedding vector(3072)
    );
    """
    cur.execute(create_table_query)

    print(f"Generando embeddings para {len(productos)} productos con Gemini...")
    client = genai.Client()
    
    for i, p in enumerate(productos):
        # Si ya tiene embedding en el JSON, lo omitimos para no gastar API
        if p.get("embedding"):
            continue
            
        texto = f"{p.get('nombre', '')}\n{p.get('descripcion', '')}"
        try:
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=texto
            )
            p["embedding"] = result.embeddings[0].values
        except Exception as e:
            print(f"Error generando embedding para {p.get('nombre')}: {e}")
            
        if (i + 1) % 10 == 0:
            print(f"Generados {i + 1}/{len(productos)}...")

    # Guardar los embeddings de vuelta en el archivo JSON
    print("Guardando embeddings en el archivo JSON...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Insertar datos
    print("Insertando datos en la base de datos...")
    insert_query = """
    INSERT INTO products (
        nombre, precio_clp, url, descripcion, disponibilidad,
        precio_original_clp, en_oferta, descuento_pct, agotado, embedding
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """

    for p in productos:
        cur.execute(insert_query, (
            p.get("nombre"),
            p.get("precio_clp"),
            p.get("url"),
            p.get("descripcion"),
            p.get("disponibilidad"),
            p.get("precio_original_clp"),
            p.get("en_oferta", False),
            p.get("descuento_pct"),
            p.get("agotado", False),
            p.get("embedding")
        ))

    conn.commit()
    cur.close()
    conn.close()

    print("¡Exito! Los productos y sus vectores fueron insertados correctamente.")

if __name__ == "__main__":
    main()
