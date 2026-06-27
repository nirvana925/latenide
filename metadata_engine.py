import os
import cv2
import json
import numpy as np
import rawpy
from pydantic import BaseModel, Field
from typing import List

# Import the unified Google GenAI SDK
from google import genai
from google.genai import types

# Define your absolute schema using Pydantic
class PhotoMetadata(BaseModel):
    title: str = Field(description="An evocative, artistic title for the image.")
    composition_style: str = Field(description="The structural composition rules used (e.g., Rule of Thirds, Leading Lines, Minimalist).")
    focal_criteria: str = Field(description="Describe what is sharp vs what is out of focus or bokeh.")
    mood_profile: str = Field(description="3-4 emotional keywords describing the aesthetic vibe.")
    color_palette: List[str] = Field(description="3 hex color codes representing the dominant colors.")
    seo_keywords: List[str] = Field(description="10 search tags relevant to this image for index tagging.")

def generate_web_preview(file_path, output_dir, quality=80):
    """Extracts the embedded thumbnail and scales it down to keep token costs near $0."""
    filename = os.path.basename(file_path)
    preview_path = os.path.join(output_dir, f"web_{os.path.splitext(filename)[0]}.jpg")
    
    with rawpy.imread(file_path) as raw:
        thumb = raw.extract_thumb()
        img_array = np.frombuffer(thumb.data, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        h, w = img.shape[:2]
        max_dim = 1024
        if w > max_dim or h > max_dim:
            scale = max_dim / max(w, h)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            
        cv2.imwrite(preview_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        return preview_path

def analyze_image_metadata(preview_jpg_path):
    """Ships the preview to the model and mandates a strict JSON response via Pydantic."""
    if not os.environ.get("GEMINI_API_KEY"):
        print(" ❌ Error: GEMINI_API_KEY environment variable not found.")
        return None

    try:
        # Initialize the client
        client = genai.Client()
        
        # # Upload the temporary web preview using the Files API
        # image_asset = client.files.upload(file=preview_jpg_path)
        
        # prompt = "Analyze this photography asset as an expert creative assistant and populate the requested schema."
        
        # # Swapped to 'gemini-2.5-flash' to use the live API gateway
        # response = client.models.generate_content(
        #     model='gemini-2.5-flash',
        #     contents=[image_asset, prompt],
        #     config=types.GenerateContentConfig(
        #         response_mime_type="application/json",
        #         response_schema=PhotoMetadata,
        #     ),
        # )
        
        # # Clean up the cloud storage asset file immediately
        # client.files.delete(name=image_asset.name)
        

        # Read the compressed preview JPEG directly from disk into memory as bytes
        with open(preview_jpg_path, "rb") as f:
            image_bytes = f.read()

        # Wrap the bytes into an ultra-fast inline data part
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg"
        )
        
        prompt = "Analyze this photography asset as an expert creative assistant and populate the requested schema."
        
        # Execute content generation inline—cutting out 2 network round-trips entirely
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[image_part, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PhotoMetadata,
            ),
        )
        
        return json.loads(response.text)

    except Exception as e:
        print(f" ❌ API Analysis Failure: {e}")
        return None