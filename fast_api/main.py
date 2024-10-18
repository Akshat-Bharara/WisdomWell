from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from typing import Dict, Any
import os
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pathlib import Path
import re  
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
API_KEY = os.getenv('API_KEY')
genai.configure(api_key=API_KEY)

model = genai.GenerativeModel("gemini-1.5-flash")

# Generate prompt for knowledge tree
def generate_prompt(file_type: str, tree_json: Dict[str, Any]):
    return (
        f"You are an AI tasked with building an intelligent and well-structured knowledge tree for study purposes. "
        f"The current knowledge tree contains the following information: {tree_json}. Your task is to expand and "
        f"refine this tree by analyzing the newly provided {file_type}. Please ensure the following while updating the tree: \n"
        "1. Break down the content into well-defined nodes (concepts) and edges (relationships between concepts).\n"
        "2. Make sure each node is associated with relevant subtopics, key points, and summaries.\n"
        "3. For any academic content, include references to important sections, diagrams, or formulas that enhance understanding.\n"
        "4. Ensure the tree is organized logically, so each node flows into the next in a manner that promotes a clear learning pathway.\n"
        "5. Capture key insights, examples, and practical applications wherever applicable.\n"
        "6. If the file is an image, detect and describe the main elements, charts, or text visible in the image, and integrate them into the tree meaningfully.\n"
        "7. If the file is a PDF, parse the document and extract essential sections, topics, and summaries to form nodes, linked with related topics in the tree.\n"
        r"8. The tree should be in the format of a JSON object representing a hierarchical structure with nodes and edges. "
        r"9. There should not be any \n or \t characters in the JSON object. "
        r"The schema for the knowledge tree is as follows: {root: {id: string, label: string, subtopics: List[string], key_points: List[string], summary: string, children: List[root]}}"
        "The id should be a unique identifier for each node, and the source and target in the edges should correspond to the node ids."
        "And the id should describe the node in a concise and meaningful way."
    )

# Clean and parse the JSON response from Gemini
def clean_and_parse_response(response_text: str):
    # Remove newlines and any extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', response_text).strip()
    
    # Find the JSON object in the cleaned text
    json_match = re.search(r'\{.*\}', cleaned_text)
    if json_match:
        json_str = json_match.group(0)
        try:
            # Parse the JSON string
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
    else:
        print("No valid JSON object found in the response")
        return None

class TreeData(BaseModel):
    tree_json: Dict[str, Any]

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...), tree_json: str = Form(...)):
    pdf_path = None
    try:
        tree_data = TreeData(tree_json=json.loads(tree_json))
        
        pdf_path = Path(f"uploads/{file.filename}")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_pdf = genai.upload_file(pdf_path)
        prompt = generate_prompt("PDF", tree_data.tree_json)
        response = model.generate_content([prompt, uploaded_pdf])
        new_tree = response.text
        
        cleaned_tree_data = clean_and_parse_response(new_tree)
        if cleaned_tree_data:
            return cleaned_tree_data
        else:
            # If parsing fails, return the raw response for debugging
            return {"error": "Failed to parse the tree data", "raw_response": new_tree}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if pdf_path and pdf_path.exists():
            pdf_path.unlink()

# Add image upload handling
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), tree_json: str = Form(...)):
    image_path = None
    try:
        tree_data = TreeData(tree_json=json.loads(tree_json))
        
        # Save the image file to the uploads directory
        image_path = Path(f"uploads/{file.filename}")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_image = genai.upload_file(image_path)
        prompt = generate_prompt("Image", tree_data.tree_json)
        response = model.generate_content([prompt, uploaded_image])
        new_tree = response.text
        
        cleaned_tree_data = clean_and_parse_response(new_tree)
        if cleaned_tree_data:
            return cleaned_tree_data
        else:
            # If parsing fails, return the raw response for debugging
            return {"error": "Failed to parse the tree data", "raw_response": new_tree}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if image_path and image_path.exists():
            image_path.unlink()

# Handle YouTube video input
@app.post("/upload/youtube")
async def upload_youtube(video_url: str = Form(...), tree_json: str = Form(...)):
    try:
        tree_data = TreeData(tree_json=json.loads(tree_json))
        
        prompt = generate_prompt("YouTube video: "+ video_url, tree_data.tree_json)
        response = model.generate_content([prompt])
        new_tree = response.text
        
        cleaned_tree_data = clean_and_parse_response(new_tree)
        if cleaned_tree_data:
            return cleaned_tree_data
        else:
            # If parsing fails, return the raw response for debugging
            return {"error": "Failed to parse the tree data", "raw_response": new_tree}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
