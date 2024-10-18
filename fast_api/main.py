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

def generate_prompt(file_type: str, graph_json: Dict[str, Any]):
    return (
        f"You are an AI tasked with building an intelligent and well-structured knowledge graph for study purposes. "
        f"The current knowledge graph contains the following information: {graph_json}. Your task is to expand and "
        f"refine this graph by analyzing the newly provided {file_type}. Please ensure the following while updating the graph: \n"
        "1. Break down the content into well-defined nodes (concepts) and edges (relationships between concepts).\n"
        "2. Make sure each node is associated with relevant subtopics, key points, and summaries.\n"
        "3. For any academic content, include references to important sections, diagrams, or formulas that enhance understanding.\n"
        "4. Ensure the graph is organized logically, so each node flows into the next in a manner that promotes a clear learning pathway.\n"
        "5. Capture key insights, examples, and practical applications wherever applicable.\n"
        "6. If the file is an image, detect and describe the main elements, charts, or text visible in the image, and integrate them into the graph meaningfully.\n"
        "7. If the file is a PDF, parse the document and extract essential sections, topics, and summaries to form nodes, linked with related topics in the graph.\n"
        "8. Ensure all nodes and edges are clearly labeled to maintain clarity in the study graph structure.\n"
        "9. Do not include any \n in the final response."
        "Please proceed with this task and return the updated JSON structure of the knowledge graph."
    )

def clean_and_parse_response(response_text: str):
    cleaned_graph = re.sub(r'```json\s*|\s*```', '', response_text).strip()

    try:
        return json.loads(cleaned_graph)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None

@app.post("/upload/pdf")
async def upload_pdf(file: UploadFile = File(...), graph_json: str = Form(...)):
    try:
        graph_data = GraphData(graph_json=eval(graph_json)) 
        
        pdf_path = Path(f"uploads/{file.filename}")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        with pdf_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_pdf = genai.upload_file(pdf_path)
        print(f"PDF uploaded: {uploaded_pdf}")
        
        prompt = generate_prompt("PDF", graph_data.graph_json)
        
        response = model.generate_content([prompt, uploaded_pdf])
        new_graph = response.text

        cleaned_graph_data = clean_and_parse_response(new_graph)
        
        if cleaned_graph_data:
            return {"message": "PDF processed successfully", "new_graph": cleaned_graph_data}
        else:
            raise HTTPException(status_code=500, detail="Failed to parse the graph data")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...), graph_json: str = Form(...)):
    try:
        graph_data = GraphData(graph_json=eval(graph_json))  
        
        image_path = Path(f"uploads/{file.filename}")
        image_path.parent.mkdir(parents=True, exist_ok=True)
        with image_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        uploaded_image = genai.upload_file(image_path)
        print(f"Image uploaded: {uploaded_image}")
        
        prompt = generate_prompt("Image", graph_data.graph_json)
        
        response = model.generate_content([prompt, uploaded_image])
        new_graph = response.text

        cleaned_graph_data = clean_and_parse_response(new_graph)
        
        if cleaned_graph_data:
            return {"message": "Image processed successfully", "new_graph": cleaned_graph_data}
        else:
            raise HTTPException(status_code=500, detail="Failed to parse the graph data")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/update/graph")
async def update_graph(graph_json: str = Form(...)):
    try:
        graph_data = GraphData(graph_json=eval(graph_json))  
        
        prompt = generate_enhancement_prompt(graph_data.graph_json)
        
        response = model.generate_content([prompt])
        updated_graph = response.text

        cleaned_graph_data = clean_and_parse_response(updated_graph)

        if cleaned_graph_data:
            return {"message": "Knowledge graph updated successfully", "updated_graph": cleaned_graph_data}
        else:
            raise HTTPException(status_code=500, detail="Failed to parse the updated graph data")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def generate_enhancement_prompt(existing_graph: Dict[str, Any]):
    return (
        f"You are an AI assistant tasked with automatically enhancing a knowledge graph for study purposes."
        f"The current knowledge graph contains the following information: {existing_graph}. "
        "Your task is to intelligently refine and optimize this graph structure by:\n"
        "1. Identifying any gaps or missing concepts that could enhance the understanding of the subject.\n"
        "2. Returning the updated JSON structure of the knowledge graph in the exact same json format as the existing graph.\n"
        "3. Return the response in the exact same json format as the existing graph and do not add any other extra information.\n"
        "4. The response should be in a proper json format with double quotes and not single quotes.\n"
    )

class GraphData(BaseModel):
    graph_json: Dict[str, Any]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
