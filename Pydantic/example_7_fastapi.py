import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pydantic_ai import Agent

# Aquí simularemos un Agente básico para el ejemplo. 
# En tu proyecto real, importarías el agente desde tu archivo (ej: from example_6 import agent, DbDependencies)

agent = Agent(
    model="google-gla:gemini-1.5-flash",
    system_prompt="Eres un asistente muy útil. Responde a las preguntas de manera clara."
)

app = FastAPI(title="Mi Agente con Streaming")

class ChatRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    """
    Este endpoint recibe la consulta y devuelve el texto palabra por palabra.
    """
    async def generar_respuesta():
        # run_stream abre un contexto para recibir los chunks de la respuesta
        async with agent.run_stream(request.query) as result:
            async for chunk in result.stream_text():
                # Emitimos cada pedazo de texto a medida que el LLM lo genera
                yield chunk

    # Retornamos un StreamingResponse que mantiene la conexión abierta
    return StreamingResponse(
        generar_respuesta(), 
        media_type="text/plain"
    )

# Para correr este archivo:
# 1. Instala las dependencias: pip install fastapi uvicorn
# 2. Levanta el servidor: uvicorn example_7_fastapi:app --reload
# 3. Prueba con curl (o Postman):
#    curl -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"query": "Dime un resumen de 5 lineas sobre agricultura."}'
