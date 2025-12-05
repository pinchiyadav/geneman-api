#!/usr/bin/env python3
"""
GeneMAN API Server
Generalizable single-image 3D human reconstruction
"""

import os
import uuid
from io import BytesIO
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional
from PIL import Image
import uvicorn

app = FastAPI(
    title="GeneMAN API",
    description="Generalizable single-image 3D human reconstruction",
    version="1.0.0"
)

class GenerationRequest(BaseModel):
    image_url: Optional[str] = Field(None, description="URL of the input image")
    quality: str = Field("high", description="Quality preset: low, medium, high")

class GenerationResponse(BaseModel):
    success: bool
    message: str
    output_path: Optional[str] = None

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate/upload", response_model=GenerationResponse)
async def generate_3d_upload(
    file: UploadFile = File(...),
    quality: str = Query("high", description="Quality: low, medium, high")
):
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGBA")
        
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save input for processing
        input_path = os.path.join(output_dir, f"{uuid.uuid4()}_input.png")
        image.save(input_path)
        
        # Note: Full GeneMAN pipeline requires preprocessing and multi-stage generation
        # This is a simplified placeholder
        
        return GenerationResponse(
            success=True,
            message="Image saved. Run GeneMAN pipeline for full reconstruction.",
            output_path=input_path
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join("outputs", filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=filename)

def main():
    uvicorn.run(app, host="0.0.0.0", port=8003)

if __name__ == "__main__":
    main()
