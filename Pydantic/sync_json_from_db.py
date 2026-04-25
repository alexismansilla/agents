import json
import psycopg2

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

    json_path = "data/fertilizacion.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    productos = data.get("productos", [])
    
    print("Sincronizando embeddings desde la base de datos al JSON...")
    actualizados = 0
    for p in productos:
        cur.execute("SELECT embedding::text FROM products WHERE nombre = %s;", (p.get("nombre"),))
        row = cur.fetchone()
        if row and row[0]:
            # Convertimos de texto "[0.123, 0.456]" a lista real
            # row[0] de pgvector text mode devuelve un string "[...]"
            p["embedding"] = json.loads(row[0])
            actualizados += 1
            
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        
    print(f"¡Sincronización completa! Se actualizaron {actualizados} embeddings en el archivo JSON.")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
