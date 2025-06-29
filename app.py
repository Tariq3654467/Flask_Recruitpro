from flask import Flask, request, jsonify, send_file, session
import os
import json
import pandas as pd
from docx import Document
import pdfplumber
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import uuid
import io
import secrets
import time
import logging

app = Flask(__name__, static_folder=os.path.abspath('static'))
app.secret_key = secrets.token_hex(16)  # For session management

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Groq client
client = Groq(api_key="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")

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_size = embedding_model.get_sentence_embedding_dimension()

# Global dictionary to store vector stores by session ID
vector_stores = {}

def get_vector_store():
    """Get or create vector store for current session"""
    session_id = session.get('id')
    if not session_id:
        session_id = str(uuid.uuid4())
        session['id'] = session_id
        vector_stores[session_id] = {
            'index': faiss.IndexFlatIP(embedding_size),
            'metadata': [],
            'resume_count': 0,
            'last_processed': 0
        }
    
    if session_id not in vector_stores:
        vector_stores[session_id] = {
            'index': faiss.IndexFlatIP(embedding_size),
            'metadata': [],
            'resume_count': 0,
            'last_processed': 0
        }
    
    return vector_stores[session_id]

# Helper Functions
def extract_text_from_pdf(file):
    text = ""
    try:
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return str(e), False
    return text, True

def extract_text_from_docx(file):
    text = ""
    try:
        doc = Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        logger.error(f"Error extracting text from DOCX: {str(e)}")
        return str(e), False
    return text, True

def extract_resume_data(text):
    prompt = f"""
    Extract the following information from the resume below in JSON format with keys: 
    "name", "email", "phone", "skills" (list), "experience" (list of dicts with "title", "company", "duration", "description"), 
    "education" (list of dicts with "degree", "institution", "year"), "certifications" (list), "summary".
    
    Resume:
    {text[:15000]}
    
    Return ONLY the JSON object, nothing else.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        return json.loads(chat_completion.choices[0].message.content), True
    except Exception as e:
        logger.error(f"Error extracting resume data: {str(e)}")
        return {"error": str(e)}, False

def analyze_candidate_fit(candidate_data, job_description):
    prompt = f"""
    Analyze candidate fit for the role. Return JSON with:
    - score (0-100)
    - strengths (list)
    - red_flags (list)
    - justification
    
    Candidate: {json.dumps(candidate_data)}
    Job Description: {job_description}
    
    Return ONLY the JSON object, nothing else.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            response_format={"type": "json_object"},
            max_tokens=3000
        )
        return json.loads(chat_completion.choices[0].message.content), True
    except Exception as e:
        logger.error(f"Error analyzing candidate fit: {str(e)}")
        return {"error": str(e)}, False

def generate_report(candidates, job_desc):
    if not candidates or not job_desc:
        return "No candidates or job description provided.", False
    
    serializable_candidates = [
        {k: v for k, v in c.items() if isinstance(v, (str, int, float, list, dict))}
        for c in candidates[:10]
    ]
    prompt = f"""
    Generate a comprehensive recruitment report comparing candidates for a job role.
    Include candidate rankings, strengths, weaknesses, and recommendations.
    
    Job Description:
    {job_desc}
    
    Candidates:
    {json.dumps(serializable_candidates)}
    
    Structure your report with:
    1. Executive Summary
    2. Candidate Comparison (table format)
    3. Detailed Analysis per Candidate
    4. Final Recommendations
    
    Return the report in Markdown format.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=4000
        )
        return chat_completion.choices[0].message.content, True
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return str(e), False

def answer_recruitment_question(query, candidates, job_desc):
    if not query or not candidates or not job_desc:
        return "Query, candidates, and job description are required.", False
    
    serializable_candidates = [
        {k: v for k, v in c.items() if isinstance(v, (str, int, float, list, dict))}
        for c in candidates[:10]
    ]
    prompt = f"""
    You are an expert recruitment assistant. Answer the following question based on the candidates and job description.
    
    Question: {query}
    
    Job Description:
    {job_desc}
    
    Candidates:
    {json.dumps(serializable_candidates)}
    
    Provide a concise, accurate answer with relevant details.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content, True
    except Exception as e:
        logger.error(f"Error answering question: {str(e)}")
        return str(e), False

# API Endpoints
@app.route('/')
def index():
    try:
        return app.send_static_file('index.html')
    except Exception as e:
        logger.error(f"Error loading index.html: {str(e)}")
        return f"Error loading index.html: {str(e)}", 500

@app.route('/debug')
def debug():
    return jsonify({
        "static_folder": app.static_folder,
        "files": os.listdir(app.static_folder) if os.path.exists(app.static_folder) else "NOT FOUND"
    })

@app.route('/api/init_vector_db', methods=['POST'])
def init_vector_db():
    vector_store = get_vector_store()
    vector_store['index'] = faiss.IndexFlatIP(embedding_size)
    vector_store['metadata'] = []
    vector_store['resume_count'] = 0
    vector_store['last_processed'] = time.time()
    return jsonify({"status": "Vector DB initialized", "resume_count": 0})

@app.route('/api/vector_db_status', methods=['GET'])
def vector_db_status():
    vector_store = get_vector_store()
    return jsonify({
        "initialized": vector_store['index'] is not None,
        "resume_count": vector_store['resume_count']
    })

@app.route('/api/process_resumes', methods=['POST'])
def process_resumes():
    if 'job_description' not in request.form or not request.files.getlist('resumes'):
        return jsonify({"error": "Job description and resumes are required"}), 400
    
    job_desc = request.form['job_description']
    rag_mode = request.form.get('rag_mode', 'false').lower() == 'true'
    top_n = int(request.form.get('top_n', 5))
    processed_candidates = []
    vector_store = get_vector_store()
    
    # Reset vector DB if it's been more than 1 hour since last use
    if time.time() - vector_store.get('last_processed', 0) > 3600:
        vector_store['index'] = faiss.IndexFlatIP(embedding_size)
        vector_store['metadata'] = []
        vector_store['resume_count'] = 0
    
    vector_store['last_processed'] = time.time()
    
    if rag_mode:
        embeddings = []
        metadata = []
        
        for file in request.files.getlist('resumes'):
            if file.mimetype == "application/pdf":
                text, success = extract_text_from_pdf(file)
            else:
                text, success = extract_text_from_docx(file)
                
            if not success:
                continue
                
            # Check if this resume is already in the system
            existing = next((m for m in vector_store['metadata'] if m['filename'] == file.filename and m['text'] == text[:10000]), None)
            if existing:
                continue
                
            embedding = embedding_model.encode(text)
            embeddings.append(embedding)
            metadata.append({
                "filename": file.filename,
                "text": text[:10000],
                "id": str(uuid.uuid4())
            })
            vector_store['resume_count'] += 1
        
        if embeddings:
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Update vector store
            if vector_store['index'] is None:
                vector_store['index'] = faiss.IndexFlatIP(embedding_size)
            
            vector_store['index'].add(embeddings_array)
            vector_store['metadata'].extend(metadata)
            
            # Perform similarity search
            query_embedding = embedding_model.encode([job_desc])
            query_embedding = np.array(query_embedding).astype('float32')
            distances, indices = vector_store['index'].search(query_embedding, min(top_n, vector_store['index'].ntotal))
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < 0 or idx >= len(vector_store['metadata']):
                    continue
                resume_data = vector_store['metadata'][idx]
                parsed_data, success = extract_resume_data(resume_data['text'])
                if success:
                    analysis_data, success = analyze_candidate_fit(parsed_data, job_desc)
                    if success:
                        candidate_data = {
                            **parsed_data,
                            **analysis_data,
                            "filename": resume_data['filename'],
                            "id": resume_data['id'],
                            "initial_similarity": float(distance)
                        }
                        processed_candidates.append(candidate_data)
    else:
        for file in request.files.getlist('resumes'):
            if file.mimetype == "application/pdf":
                text, success = extract_text_from_pdf(file)
            else:
                text, success = extract_text_from_docx(file)
                
            if not success:
                continue
                
            parsed_data, success = extract_resume_data(text)
            if success:
                analysis_data, success = analyze_candidate_fit(parsed_data, job_desc)
                if success:
                    candidate_data = {
                        **parsed_data,
                        **analysis_data,
                        "filename": file.filename,
                        "id": str(uuid.uuid4())
                    }
                    processed_candidates.append(candidate_data)
    
    processed_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)
    return jsonify(processed_candidates)

@app.route('/api/generate_report', methods=['POST'])
def generate_report_endpoint():
    data = request.get_json()
    candidates = data.get('candidates', [])
    job_desc = data.get('job_description', '')
    
    logger.debug(f"Received generate_report request: candidates={len(candidates)}, job_desc={job_desc[:100]}...")
    
    if not candidates:
        logger.error("No candidates provided")
        return jsonify({"error": "No candidates provided"}), 400
    if not job_desc:
        logger.error("No job description provided")
        return jsonify({"error": "No job description provided"}), 400
    
    report, success = generate_report(candidates, job_desc)
    if success:
        logger.debug("Report generated successfully")
        return jsonify({"report": report})
    logger.error(f"Failed to generate report: {report}")
    return jsonify({"error": f"Failed to generate report: {report}"}), 500

@app.route('/api/ask_question', methods=['POST'])
def ask_question():
    data = request.get_json()
    query = data.get('query', '')
    candidates = data.get('candidates', [])
    job_desc = data.get('job_description', '')
    
    if not query or not candidates or not job_desc:
        return jsonify({"error": "Query, candidates, and job description are required"}), 400
    
    response, success = answer_recruitment_question(query, candidates, job_desc)
    if success:
        return jsonify({"response": response})
    return jsonify({"error": response}), 500

@app.route('/api/export_csv', methods=['POST'])
def export_csv():
    data = request.get_json()
    candidates = data.get('candidates', [])
    
    if not candidates:
        return jsonify({"error": "No candidates data"}), 400
        
    export_data = []
    for candidate in candidates:
        export_data.append({
            "Name": candidate.get('name', ''),
            "Email": candidate.get('email', ''),
            "Phone": candidate.get('phone', ''),
            "Score": candidate.get('score', 0),
            "Top Skills": ", ".join(candidate.get('skills', [])[:5]),
            "Experience Years": len(candidate.get('experience', [])),
            "Education": ", ".join([edu.get('degree', '') for edu in candidate.get('education', [])]),
            "Strengths": "; ".join(candidate.get('strengths', [])),
            "Red Flags": "; ".join(candidate.get('red_flags', []))
        })
    
    df = pd.DataFrame(export_data)
    output = io.StringIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name='candidate_ranking.csv'
    )

if __name__ == '__main__':
    app.run(debug=True)