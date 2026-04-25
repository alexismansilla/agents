import os
import psycopg2
from google import genai
from dotenv import load_dotenv

load_dotenv()
if 'GEMINI_API_KEY' in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']

def main():
    print("Conectando a PostgreSQL...")
    conn = psycopg2.connect(
        dbname="agents",
        user="myuser",
        password="mypassword",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    # Seleccionar los que no tienen embedding
    cur.execute("SELECT id, nombre, descripcion FROM products WHERE embedding IS NULL;")
    rows = cur.fetchall()

    if not rows:
        print("Todos los productos ya tienen embeddings.")
        return

    print(f"Generando embeddings para {len(rows)} productos...")
    client = genai.Client()

    for idx, (prod_id, nombre, desc) in enumerate(rows, 1):
        texto = f"{nombre}\n{desc}"
        try:
            # Generamos de a 1 para asegurar que no se pierdan
            result = client.models.embed_content(
                model="gemini-embedding-2",
                contents=texto
            )
            embedding = result.embeddings[0].values

            # Actualizamos la DB
            cur.execute(
                "UPDATE products SET embedding = %s::vector WHERE id = %s",
                (embedding, prod_id)
            )
            
            if idx % 10 == 0:
                print(f"Procesados {idx}/{len(rows)}...")
                conn.commit()  # commit cada 10 para no perder avance
                
        except Exception as e:
            print(f"Error procesando ID {prod_id}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    print("¡Listo! Todos los embeddings vacíos han sido actualizados.")

if __name__ == "__main__":
    main()
