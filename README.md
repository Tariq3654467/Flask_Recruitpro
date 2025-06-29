RecruitAI - Intelligent Recruitment Assistant
Welcome to RecruitAI, an AI-powered recruitment assistant designed to streamline the hiring process. This application allows you to upload resumes and job descriptions, analyze candidate qualifications, generate detailed reports, and interact with an AI chatbot for recruitment insights.
Features

Resume Parsing: Extract key details (name, skills, experience, etc.) from PDF and DOCX resumes.
Candidate Ranking: Analyze candidate fit based on job descriptions using AI models.
Report Generation: Create comprehensive Markdown reports comparing candidates.
Chat Assistant: Ask questions about candidates and receive AI-powered responses.
Vector Database: Utilize FAISS for efficient similarity search with RAG (Retrieval-Augmented Generation) mode.
Analytics: View insights on technical skills, experience levels, and cultural fit.

Requirements

Python 3.8+
Flask
Groq API Key (for AI processing)
sentence-transformers
faiss-cpu or faiss-gpu (depending on your setup)
pdfplumber
python-docx
pandas
numpy

Installation
Prerequisites

Ensure Python 3.8+ is installed on your system.
Install the required Python packages:pip install flask groq sentence-transformers faiss-cpu pdfplumber python-docx pandas numpy


Use faiss-gpu instead of faiss-cpu if you have a compatible GPU.



Setup

Clone the repository:
git clone https://github.com/yourusername/recruitai.git
cd recruitai


Create a static folder in the project root and place the index.html file inside it (as generated earlier).

Obtain a Groq API key from Groq and replace the placeholder in the Flask app:
client = Groq(api_key="your_groq_api_key_here")


Run the application:
python app.py


The app will start on http://localhost:5000 by default.



Usage

Upload Resumes and Job Description:

Navigate to the "Home" page.
Drag and drop PDF/DOCX resumes or use the "Browse Files" button.
Enter a job description in the provided textarea.
Click "Process Resumes" to analyze candidates.


View Candidates:

Go to the "Candidates" page to see ranked candidates with scores and details.
Use the search bar to filter candidates.


Interact with Chat:

Visit the "Chat" page and ask questions (e.g., "Who has the most experience with Python?").


Generate Reports:

Navigate to the "Reports" page.
Select candidates and enter a report title.
Click "Generate Report" to preview and download the report.


Configure Settings:

Use the "Settings" page to adjust AI model selection, RAG settings, and database options.



Project Structure

app.py: Main Flask application with API endpoints.
static/index.html: Frontend HTML file with embedded JavaScript and CSS.
requirements.txt: All dependencies in this file

README.md: This documentation file.

API Endpoints

GET /: Serves the index.html file.
POST /api/process_resumes: Processes uploaded resumes.
POST /api/generate_report: Generates a recruitment report.
POST /api/ask_question: Handles chat queries.
POST /api/init_vector_db: Initializes the vector database.
GET /api/vector_db_status: Returns vector database status.
POST /api/export_csv: Exports candidate data to CSV.

Contributing
Contributions are welcome! Please fork the repository and submit pull requests with your changes. Ensure to:

Follow the existing code style.
Add tests or documentation for new features.
Update the README if necessary.


Powered by Groq for AI capabilities.
Utilizes SentenceTransformers and FAISS for embeddings and similarity search.
Built with Flask and Bootstrap.
