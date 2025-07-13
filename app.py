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
import smtplib
from email.mime.text import MIMEText
from dotenv import load_dotenv
import sqlite3
from urllib.parse import quote
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__, static_folder=os.path.abspath('static'))
app.secret_key = secrets.token_hex(16)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# SMTP Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_size = embedding_model.get_sentence_embedding_dimension()

# Global dictionary to store vector stores by session ID
vector_stores = {}

# Database setup
def get_db_connection():
    conn = sqlite3.connect('interviews.db', check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db_connection() as conn:
        conn.execute('DROP TABLE IF EXISTS interviews')
        conn.execute('''CREATE TABLE interviews
                        (id TEXT PRIMARY KEY, candidate_email TEXT, candidate_name TEXT, questions TEXT,
                         responses TEXT, agent_questions TEXT, job_desc TEXT, summary TEXT, created_at TIMESTAMP)''')
        conn.commit()

init_db()

def get_vector_store():
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
            max_tokens=2000
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

def generate_job_description(key_terms):
    prompt = f"""
    Generate a professional job description based on the following key terms. Include sections for Job Title, Overview, Responsibilities, Qualifications, and Preferred Skills.

    Key Terms: {', '.join(key_terms)}

    Return the job description in Markdown format.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-70b-8192",
            temperature=0.4,
            max_tokens=2000
        )
        return chat_completion.choices[0].message.content, True
    except Exception as e:
        logger.error(f"Error generating job description: {str(e)}")
        return str(e), False

def generate_follow_up_question(interview_id, candidate_response):
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return "Invalid interview session", False
        questions = json.loads(interview['questions'])
        responses = json.loads(interview['responses'])
        job_desc = interview['job_desc']
        candidate_name = interview['candidate_name']
        prompt = f"""
        You are an expert recruitment interviewer. Based on the candidate's latest response, generate a relevant follow-up question to assess their suitability for the role. The question should be specific, relevant to the job description, and aim to elicit detailed insights into the candidate's skills, experience, or fit.

        Candidate Name: {candidate_name}
        Job Description: {job_desc}
        Previous Questions: {json.dumps(questions)}
        Previous Responses: {json.dumps(responses)}
        Latest Candidate Response: {candidate_response}

        Return ONLY the follow-up question as a string.
        """
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="gemma2-9b-it",
                temperature=0.3,
                max_tokens=200
            )
            question = chat_completion.choices[0].message.content.strip()
            questions.append(question)
            agent_questions = json.loads(interview['agent_questions'])
            agent_questions.append(question)
            conn.execute('UPDATE interviews SET questions = ?, agent_questions = ? WHERE id = ?',
                        (json.dumps(questions), json.dumps(agent_questions), interview_id))
            conn.commit()
            return question, True
        except Exception as e:
            logger.error(f"Error generating follow-up question: {str(e)}")
            return str(e), False

def generate_interview_summary(interview_id):
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return "Invalid interview session", False
        candidate_name = interview['candidate_name']
        job_desc = interview['job_desc']
        questions = json.loads(interview['questions'])
        responses = json.loads(interview['responses'])
        prompt = f"""
        You are an expert recruitment assistant. Generate a summary of the candidate's interview for the hiring team. Include:
        - Candidate Name
        - Overview of Performance
        - Key Strengths
        - Areas for Improvement
        - Suitability for the Role
        - Recommendation

        Job Description: {job_desc}
        Questions Asked: {json.dumps(questions)}
        Candidate Responses: {json.dumps(responses)}

        Return the summary in Markdown format.
        """
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-70b-8192",
                temperature=0.4,
                max_tokens=2000
            )
            summary = chat_completion.choices[0].message.content
            conn.execute('UPDATE interviews SET summary = ? WHERE id = ?', (summary, interview_id))
            conn.commit()
            return summary, True
        except Exception as e:
            logger.error(f"Error generating interview summary: {str(e)}")
            return str(e), False

def send_interview_summary(recipient_email, candidate_name, summary):
    sender_email = "thinkrecruit1@gmail.com"
    sender_password = "hhjqhktxqkaviyhe"
    try:
        if not all([recipient_email, sender_email, sender_password]) or '@' not in recipient_email or '@' not in sender_email:
            logger.error(f"Invalid email address: {recipient_email} or {sender_email}")
            return False
        msg = MIMEText(f"Dear Hiring Team,\n\nPlease find the interview summary for {candidate_name} below:\n\n{summary}\n\nBest regards,\nThinkcruit")
        msg['Subject'] = f"Interview Summary for {candidate_name}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info(f"Interview summary email sent to {recipient_email} for {candidate_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to send interview summary to {recipient_email}: {str(e)}")
        return False

def send_congratulatory_email(recipient_email, candidate_name):
    sender_email = "thinkrecruit1@gmail.com"
    sender_password = "hhjqhktxqkaviyhe"
    try:
        if not all([recipient_email, sender_email, sender_password]) or '@' not in recipient_email or '@' not in sender_email:
            logger.error(f"Invalid email address: {recipient_email} or {sender_email}")
            return False
        msg = MIMEText(f"""Dear {candidate_name},\n\nWe are pleased to inform you that you have been shortlisted for the position at our organization. Congratulations! We were impressed with your background and believe you will make a valuable addition to our team. Please confirm your acceptance of this offer by replying to this email at your earliest convenience.\n\nShould you have any questions or require further information, feel free to reach out.\n\nWe look forward to welcoming you aboard.\nBest regards,\nThe Hiring Team""")
        msg['Subject'] = f"Congratulations {candidate_name} - Selection Notification"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info(f"Congratulatory email sent to {recipient_email} from {sender_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send congratulatory email to {recipient_email} from {sender_email}: {str(e)}")
        return False

def send_feedback_email(recipient_email, candidate_name, feedback):
    sender_email = "thinkrecruit1@gmail.com"
    sender_password = "hhjqhktxqkaviyhe"
    try:
        if not all([recipient_email, sender_email, sender_password]) or '@' not in recipient_email or '@' not in sender_email:
            logger.error(f"Invalid email address: {recipient_email} or {sender_email}")
            return False
        if not feedback or not isinstance(feedback, str):
            feedback = "No specific feedback available. Consider improving your skills or experience based on the job description."
            logger.warning(f"Empty or invalid feedback for {candidate_name}, using default: {feedback}")
        msg = MIMEText(f"Dear {candidate_name},\n\nThank you for your application. Unfortunately, you have not been selected for this position. Below are some suggestions for improvement or reasons for non-selection:\n\n{feedback}\n\nWe encourage you to apply again in the future.\nBest regards,\nThe Hiring Team")
        msg['Subject'] = f"Feedback - Application Status for {candidate_name}"
        msg['From'] = sender_email
        msg['To'] = recipient_email
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info(f"Feedback email sent to {recipient_email} from {sender_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send feedback email to {recipient_email} from {sender_email}: {str(e)}")
        return False

def send_interview_schedule_email(recipient_email, candidate_name, schedule, interview_id):
    sender_email = "thinkrecruit1@gmail.com"
    sender_password = "hhjqhktxqkaviyhe"
    if not all([recipient_email, sender_email, sender_password, interview_id, candidate_name]):
        logger.error(f"Missing required parameters: recipient_email={recipient_email}, sender_email={sender_email}, interview_id={interview_id}, candidate_name={candidate_name}")
        return False
    if '@' not in recipient_email or '@' not in sender_email:
        logger.error(f"Invalid email address: {recipient_email} or {sender_email}")
        return False

    base_url = os.getenv('BASE_URL', 'http://localhost:5000')
    website_link = f"{base_url}/candidate_interview?interview_id={quote(interview_id)}&candidate_name={quote(candidate_name)}"
    schedule_text = schedule or f"Interview scheduled for {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time() + 3600))} PKT via chat at {website_link}."
    
    email_body = f"""Dear {candidate_name},\n\nCongratulations on being shortlisted for the position! We are excited to invite you for an interview. Details are as follows:\n\n{schedule_text}\n\nPlease join via the link above. Best regards,\nThe Hiring Team"""
    msg = MIMEText(email_body)
    msg['Subject'] = f"Interview Schedule for {candidate_name}"
    msg['From'] = sender_email
    msg['To'] = recipient_email

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        logger.info(f"Interview schedule email sent to {recipient_email} with link: {website_link}")
        return True
    except Exception as e:
        logger.error(f"Failed to send interview schedule email to {recipient_email} from {sender_email}: {str(e)}")
        return False

def make_table(table_data):
    table = Table(table_data)
    table.setStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 12)
    ])
    return table

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
    session['job_description'] = job_desc  # Store in session
    rag_mode = request.form.get('rag_mode', 'false').lower() == 'true'
    processed_candidates = []
    vector_store = get_vector_store()

    # Reset vector store if last processed is older than 1 hour
    if time.time() - vector_store.get('last_processed', 0) > 3600:
        vector_store['index'] = faiss.IndexFlatIP(embedding_size)
        vector_store['metadata'] = []
        vector_store['resume_count'] = 0
    vector_store['last_processed'] = time.time()

    files = request.files.getlist('resumes')
    if not files:
        return jsonify({"error": "No resume files provided"}), 400

    # Cache processed files based on filename and content hash
    processed_cache = session.get('processed_cache', {})
    new_files = [f for f in files if f.filename not in processed_cache or processed_cache[f.filename]['text'] != hash(f.read())]
    for f in new_files:
        f.seek(0)  # Reset file pointer after hash calculation

    if not new_files:
        logger.info("All resumes already processed, returning cached data")
        return jsonify([processed_cache[f.filename] for f in files if f.filename in processed_cache])

    # Parallel text extraction
    def extract_text(file):
        if file.mimetype == "application/pdf":
            text, success = extract_text_from_pdf(file)
        else:  # Assuming docx or msword
            text, success = extract_text_from_docx(file)
        return (file.filename, text, success) if success else (file.filename, None, False)

    with ThreadPoolExecutor(max_workers=min(4, len(new_files))) as executor:
        results = list(executor.map(extract_text, new_files))

    # Filter successful extractions
    texts = [(filename, text) for filename, text, success in results if success and text]
    if not texts:
        return jsonify({"error": "Failed to extract text from any resume"}), 500

    # Batch embeddings
    texts_to_embed = [text for _, text in texts]
    embeddings = embedding_model.encode(texts_to_embed, convert_to_numpy=True).astype('float32')
    metadata = [
        {
            "filename": filename,
            "text": text[:10000],  # Truncate to avoid memory issues
            "id": str(uuid.uuid4())
        }
        for filename, text in texts
    ]

    # Update FAISS index with batch
    if embeddings.size > 0:
        if vector_store['index'].ntotal == 0:
            vector_store['index'] = faiss.IndexFlatIP(embedding_size)
        vector_store['index'].add(embeddings)
        vector_store['metadata'].extend(metadata)
        vector_store['resume_count'] += len(embeddings)

    if rag_mode:
        # Batch analysis
        def process_candidate(idx):
            resume_text = texts[idx][1]
            parsed_data, success = extract_resume_data(resume_text)
            if not success:
                return None
            analysis_data, success = analyze_candidate_fit(parsed_data, job_desc)
            if not success:
                return None
            return {
                **parsed_data,
                **analysis_data,
                "filename": metadata[idx]["filename"],
                "id": metadata[idx]["id"],
                "initial_similarity": 0.0  # Placeholder, as similarity search is optional here
            }

        with ThreadPoolExecutor(max_workers=min(4, len(texts))) as executor:
            candidate_results = list(executor.map(process_candidate, range(len(texts))))

        processed_candidates = [c for c in candidate_results if c is not None]
    else:
        for idx, (filename, text) in enumerate(texts):
            parsed_data, success = extract_resume_data(text)
            if not success:
                continue
            analysis_data, success = analyze_candidate_fit(parsed_data, job_desc)
            if not success:
                continue
            candidate_data = {
                **parsed_data,
                **analysis_data,
                "filename": filename,
                "id": metadata[idx]["id"]
            }
            processed_candidates.append(candidate_data)

    # Update cache and sort by score
    for candidate in processed_candidates:
        processed_cache[candidate["filename"]] = {
            "text": hash(texts[[t[0] for t in texts].index(candidate["filename"])][1]),
            **candidate
        }
    session['processed_cache'] = processed_cache
    processed_candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

    logger.info(f"Processed {len(processed_candidates)} resumes at {datetime.now()}")
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

@app.route('/api/send_congratulatory_email', methods=['POST'])
def send_congratulatory_email_endpoint():
    data = request.get_json()
    logger.debug(f"Received send_congratulatory_email request: {data}")
    selected_candidates = data.get('selected_candidates', [])
    if not selected_candidates or not "thinkrecruit1@gmail.com" or not "hhjqhktxqkaviyhe":
        logger.error("Selected candidates or email credentials are missing")
        return jsonify({"error": "Selected candidates or email credentials are required"}), 400
    success_count = 0
    for candidate in selected_candidates:
        name = candidate.get('name', 'Candidate')
        email = candidate.get('email')
        if email and send_congratulatory_email(email, name):
            success_count += 1
    logger.info(f"Sent {success_count} congratulatory emails out of {len(selected_candidates)}")
    return jsonify({
        "status": "Emails sent",
        "success_count": success_count,
        "total": len(selected_candidates)
    })

@app.route('/api/send_feedback_email', methods=['POST'])
def send_feedback_email_endpoint():
    data = request.get_json()
    logger.debug(f"Received send_feedback_email request: {data}")
    unselected_candidates = data.get('unselected_candidates', [])
    if not unselected_candidates or not "thinkrecruit1@gmail.com" or not "hhjqhktxqkaviyhe":
        logger.error("Unselected candidates or email credentials are missing")
        return jsonify({"error": "Unselected candidates or email credentials are required"}), 400
    success_count = 0
    for candidate in unselected_candidates:
        name = candidate.get('name', 'Candidate')
        email = candidate.get('email')
        red_flags = candidate.get('red_flags', [])
        skills = candidate.get('skills', [])
        feedback_items = []
        if red_flags:
            feedback_items.extend(red_flags)
        if skills:
            feedback_items.append(f"Consider enhancing skills in: {', '.join(skills)}")
        feedback = "\n".join(feedback_items) if feedback_items else "No specific feedback available. Consider improving your skills or experience based on the job description."
        if email and send_feedback_email(email, name, feedback):
            success_count += 1
        else:
            logger.warning(f"Skipped sending feedback email to {name} due to invalid email or failure")
    logger.info(f"Sent {success_count} feedback emails out of {len(unselected_candidates)}")
    return jsonify({
        "status": "Emails sent",
        "success_count": success_count,
        "total": len(unselected_candidates)
    })

@app.route('/api/send_interview_schedule', methods=['POST'])
def send_interview_schedule_endpoint():
    data = request.get_json()
    logger.debug(f"Received send_interview_schedule request: {data}")
    selected_candidates = data.get('selected_candidates', [])
    schedule = data.get('schedule', "")
    if not selected_candidates or not "thinkrecruit1@gmail.com" or not "hhjqhktxqkaviyhe":
        logger.error("Selected candidates or email credentials are missing")
        return jsonify({"error": "Selected candidates or email credentials are required"}), 400
    success_count = 0
    for candidate in selected_candidates:
        name = candidate.get('name', 'Candidate')
        email = candidate.get('email')
        interview_id = str(uuid.uuid4())
        with get_db_connection() as conn:
            conn.execute('INSERT INTO interviews (id, candidate_email, candidate_name, questions, responses, agent_questions, job_desc, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (interview_id, email, name, json.dumps(["Please introduce yourself and tell us about your experience."]), json.dumps([]), json.dumps([]), data.get('job_description', ''), time.strftime('%Y-%m-%d %H:%M:%S')))
            conn.commit()
        if email and send_interview_schedule_email(email, name, schedule, interview_id):
            success_count += 1
    logger.info(f"Sent {success_count} interview schedule emails out of {len(selected_candidates)}")
    return jsonify({
        "status": "Emails sent",
        "success_count": success_count,
        "total": len(selected_candidates)
    })

@app.route('/api/generate_job_description', methods=['POST'])
def generate_job_description_endpoint():
    data = request.get_json()
    logger.debug(f"Received generate_job_description request: {data}")
    key_terms = data.get('key_terms', [])
    if not key_terms or not isinstance(key_terms, list):
        logger.error("Key terms are required as a list")
        return jsonify({"error": "Key terms are required as a list"}), 400
    job_description, success = generate_job_description(key_terms)
    if success:
        logger.debug("Job description generated successfully")
        return jsonify({"job_description": job_description})
    logger.error(f"Failed to generate job description: {job_description}")
    return jsonify({"error": f"Failed to generate job description: {job_description}"}), 500

@app.route('/api/start_interview', methods=['POST'])
def start_interview():
    data = request.get_json()
    candidate_email = data.get('candidate_email')
    candidate_name = data.get('candidate_name')
    job_desc = data.get('job_description', '')
    if not candidate_email or not candidate_name:
        return jsonify({"error": "Candidate email and name are required"}), 400
    interview_id = str(uuid.uuid4())
    with get_db_connection() as conn:
        conn.execute('INSERT INTO interviews (id, candidate_email, candidate_name, questions, responses, agent_questions, job_desc, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (interview_id, candidate_email, candidate_name, json.dumps(["Please introduce yourself and tell us about your experience."]), json.dumps([]), json.dumps([]), job_desc, time.strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()
    return jsonify({"status": "Interview started", "interview_id": interview_id, "current_question": "Please introduce yourself and tell us about your experience."})

@app.route('/api/get_interview_state', methods=['GET'])
def get_interview_state():
    interview_id = request.args.get('interview_id')
    if not interview_id:
        return jsonify({"error": "Interview ID is required"}), 400
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return jsonify({"error": "Invalid interview session"}), 404
        created_at = datetime.strptime(interview['created_at'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - created_at > timedelta(hours=24):
            summary, success = generate_interview_summary(interview_id)
            if success:
                hiring_team_email = os.getenv('HIRING_TEAM_EMAIL', 'default@example.com')
                if send_interview_summary(hiring_team_email, interview['candidate_name'], summary):
                    logger.info(f"Auto-sent summary to {hiring_team_email} for expired interview {interview_id}")
                conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
                conn.commit()
            return jsonify({"error": "Interview session has expired"}), 410
        questions = json.loads(interview['questions'])
        responses = json.loads(interview['responses'])
        agent_questions = json.loads(interview['agent_questions'])
        messages = []
        for i, question in enumerate(questions):
            messages.append({"sender": "Interviewer", "text": question})
            if i < len(responses):
                messages.append({"sender": "Candidate", "text": responses[i]})
        return jsonify({
            "status": "active" if questions else "pending",
            "messages": messages,
            "candidate_name": interview['candidate_name']
        })

@app.route('/api/submit_agent_question', methods=['POST'])
def submit_agent_question():
    data = request.get_json()
    interview_id = data.get('interview_id')
    question = data.get('question')
    if not interview_id or not question:
        return jsonify({"error": "Interview ID and question are required"}), 400
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return jsonify({"error": "Invalid interview session"}), 404
        created_at = datetime.strptime(interview['created_at'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - created_at > timedelta(hours=24):
            conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
            conn.commit()
            return jsonify({"error": "Interview session has expired"}), 410
        questions = json.loads(interview['questions'])
        agent_questions = json.loads(interview['agent_questions'])
        agent_questions.append(question)
        questions.append(question)
        conn.execute('UPDATE interviews SET questions = ?, agent_questions = ? WHERE id = ?', (json.dumps(questions), json.dumps(agent_questions), interview_id))
        conn.commit()
    return jsonify({"status": "Question submitted", "current_question": question})

@app.route('/api/submit_candidate_response', methods=['POST'])
def submit_candidate_response():
    data = request.get_json()
    interview_id = data.get('interview_id')
    response = data.get('response')
    if not interview_id or not response:
        return jsonify({"error": "Interview ID and response are required"}), 400
    if len(response.strip()) < 5:
        return jsonify({"error": "Response must be at least 5 characters long"}), 400
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return jsonify({"error": "Invalid interview session"}), 404
        created_at = datetime.strptime(interview['created_at'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - created_at > timedelta(hours=24):
            conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
            conn.commit()
            return jsonify({"error": "Interview session has expired"}), 410
        responses = json.loads(interview['responses'])
        responses.append(response.strip())
        conn.execute('UPDATE interviews SET responses = ? WHERE id = ?', (json.dumps(responses), interview_id))
        conn.commit()
    next_question, success = generate_follow_up_question(interview_id, response)
    if success:
        return jsonify({"status": "Response received", "next_question": next_question})
    return jsonify({"status": "Response received", "error": next_question}), 500

@app.route('/api/end_interview', methods=['POST'])
def end_interview():
    data = request.get_json()
    interview_id = data.get('interview_id')
    if not interview_id:
        return jsonify({"error": "Interview ID is required"}), 400
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return jsonify({"error": "Invalid interview session"}), 404
        created_at = datetime.strptime(interview['created_at'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - created_at > timedelta(hours=24):
            conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
            conn.commit()
            return jsonify({"error": "Interview session has expired"}), 410
        summary, success = generate_interview_summary(interview_id)
        if not success:
            logger.error(f"Failed to generate summary: {summary}")
            return jsonify({"error": f"Failed to generate summary: {summary}"}), 500
        hiring_team_email = os.getenv('HIRING_TEAM_EMAIL', 'thinkrecruit1@gmail.com')
        if send_interview_summary(hiring_team_email, interview['candidate_name'], summary):
            logger.info(f"Interview summary sent to {hiring_team_email}")
        else:
            logger.error(f"Failed to send summary to {hiring_team_email}")
            return jsonify({"error": "Failed to send summary to HR team"}), 500
        conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
        conn.commit()
    return jsonify({"status": "Interview ended", "summary": summary})

@app.route('/api/download_pdf', methods=['POST'])
def download_pdf():
    data = request.get_json()
    logger.debug(f"Received download_pdf request at {datetime.now()}: {data}")

    candidates = data.get('candidates', [])
    if not candidates:
        logger.error(f"No candidates data provided at {datetime.now()}")
        return jsonify({"error": "No candidates data"}), 400

    job_desc = data.get('job_description', session.get('job_description', 'No job description provided.'))
    report, success = generate_report(candidates, job_desc)
    if not success:
        logger.error(f"Failed to generate report at {datetime.now()}: {report}")
        return jsonify({"error": f"Failed to generate report: {report}"}), 500

    try:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        table_data = []
        in_table = False

        for line in report.split('\n'):
            line = line.strip()
            if not line:
                continue

            if line.startswith('# '):
                if in_table and table_data:
                    story.append(make_table(table_data))
                    table_data = []
                    in_table = False
                story.append(Paragraph(line[2:], styles['Heading1']))

            elif line.startswith('## '):
                if in_table and table_data:
                    story.append(make_table(table_data))
                    table_data = []
                    in_table = False
                story.append(Paragraph(line[3:], styles['Heading2']))

            elif line.startswith('|'):
                in_table = True
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if cells:
                    table_data.append(cells)

            else:
                if in_table and table_data:
                    story.append(make_table(table_data))
                    table_data = []
                    in_table = False
                story.append(Paragraph(line, styles['BodyText']))

            story.append(Spacer(1, 12))

        # Flush any remaining table data
        if in_table and table_data:
            story.append(make_table(table_data))

        doc.build(story)
        buffer.seek(0)
        logger.info(f"PDF generated successfully at {datetime.now()} for {len(candidates)} candidates")

        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='recruitment_report.pdf'
        )

    except Exception as e:
        logger.error(f"PDF generation failed at {datetime.now()}: {str(e)}", exc_info=True)
        return jsonify({"error": f"PDF generation failed: {str(e)}"}), 500

@app.route('/candidate_interview')
def candidate_interview():
    interview_id = request.args.get('interview_id')
    candidate_name = request.args.get('candidate_name')
    if not interview_id or not candidate_name:
        return "Invalid interview link: missing interview_id or candidate_name", 400
    with get_db_connection() as conn:
        interview = conn.execute('SELECT * FROM interviews WHERE id = ?', (interview_id,)).fetchone()
        if not interview:
            return "Invalid or expired interview link", 404
        if interview['candidate_name'].lower() != candidate_name.lower():
            return "Unauthorized: Candidate name does not match", 403
        created_at = datetime.strptime(interview['created_at'], '%Y-%m-%d %H:%M:%S')
        if datetime.now() - created_at > timedelta(hours=24):
            conn.execute('DELETE FROM interviews WHERE id = ?', (interview_id,))
            conn.commit()
            return "Interview session has expired", 410
    try:
        return app.send_static_file('candidate_interview.html')
    except Exception as e:
        logger.error(f"Error loading candidate_interview.html: {str(e)}")
        return f"Error loading interview page: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)