import os
from dotenv import load_dotenv
from pydantic_ai import Agent
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field
import asyncio

# Cargar variables de entorno desde el archivo .env
load_dotenv()

if 'GEMINI_API_KEY' in os.environ:
    os.environ['GOOGLE_API_KEY'] = os.environ['GEMINI_API_KEY']

class MaterialType(str, Enum):
    METAL = "metal"
    WOOD = "wood"
    PLASTIC = "plastic"
    GLASS = "glass"
    CERAMIC = "ceramic"
    CONCRETE = "concrete"
    PAPER = "paper"
    FABRIC = "fabric"
    OTHER = "other"


class Dimensions(BaseModel):
    width_mm: float = Field(gt=0, description="Width of the object in millimeters")
    height_mm: float = Field(gt=0, description="Height of the object in millimeters")
    depth_mm: float = Field(gt=0, description="Depth of the object in millimeters")


class SteelPiece(BaseModel):
    material: MaterialType
    dimensions: Dimensions
    weight_kg: float = Field(gt=0, description="Weight of the object in kilograms")
    quality_grade: str = Field(description="Quality grade of the object")
    observations: Optional[str] = Field(None, description="Additional observations about the object")

# MUTACIÓN 1: Ahora el output_type es una Lista de piezas (List[SteelPiece])
steel_analyst_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=List[SteelPiece],
    system_prompt=(
        "Eres un experto en ingeniería de materiales y manufactura de acero. "
        "Tu tarea es extraer especificaciones técnicas de pedidos de clientes. "
        "Si el cliente pide varios objetos, extrae cada uno de forma individual. "
        "Asegúrate de normalizar todas las unidades a milímetros y kilogramos."
    ),
) 

async def main():
    # Pedido con múltiples piezas
    raw_order = (  
        "Hola, necesito dos cosas urgentes: "
        "1. Una viga de acero inoxidable SS304 de 10cm x 50mm x 2 metros, pesa unos 15.5 kilos. "
        "2. Un panel de vidrio templado de 100cm de ancho, 2 metros de alto y 10mm de espesor, pesa 25kg."
    )
    
    print("Procesando pedidos múltiples...")

    result = await steel_analyst_agent.run(raw_order)
    piezas: List[SteelPiece] = result.output

    # MUTACIÓN 2: Iteramos sobre la respuesta como si fuera un Array de Ruby
    for i, pieza in enumerate(piezas, 1):
        print(f"\n--- Pieza {i} (JSON) ---")
        print(pieza.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(main())
