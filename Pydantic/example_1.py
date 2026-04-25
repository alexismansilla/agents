import os
from dotenv import load_dotenv
from pydantic_ai import Agent

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Pydantic AI espera GOOGLE_API_KEY para los modelos de Gemini
if 'GEMINI_API_KEY' in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']

agent = Agent("google-gla:gemini-2.5-flash-lite", instructions='Be concise, reply with one sentence.')

result = agent.run_sync('Where does "hello world" come from?')
print(result.output)
