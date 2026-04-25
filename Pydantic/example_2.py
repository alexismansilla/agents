import os
from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

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

steel_analyst_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    output_type=SteelPiece,
    system_prompt=(
        "Eres un experto en ingeniería de materiales y manufactura de acero. "
        "Tu tarea es extraer especificaciones técnicas de pedidos de clientes. "
        "Si el cliente no especifica el material, asume 'Carbon Steel' por defecto. "
        "Asegúrate de normalizar todas las unidades a milímetros y kilogramos."
    ),
) 

async def main():
    raw_order = (  
        "Hola, necesito una viga de acero inoxidable de grado SS304. "
        "Las medidas son 10cm de ancho por 50mm de alto y un largo de 2 metros. "
        "Calculamos que pesa unos 15.5 kilos. Es para un proyecto urgente." 
    )
    
    print("Procesando pedido...")

    result = await steel_analyst_agent.run(raw_order)
    data: SteelPiece = result.output

    print(f"Material: {data.material}")
    print(f"Dimensions: {data.dimensions}")
    print(f"Weight: {data.weight_kg}")
    print(f"Quality Grade: {data.quality_grade}")
    print(f"Observations: {data.observations}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
        