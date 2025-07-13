Recruitment Agent
The Recruitment Agent is a powerful Flask-based application designed to streamline the recruitment process. It leverages advanced AI models (via the Groq API) and natural language processing (using SentenceTransformer and FAISS) to process resumes, analyze candidate fit, generate reports, conduct interviews, and manage email communications. This tool is ideal for HR professionals and recruiters looking to automate and enhance candidate evaluation.

Features
Resume Processing: Extract and analyze data from PDF and DOCX resumes, including skills, experience, education, and more.
Candidate Fit Analysis: Evaluate candidate suitability based on job descriptions with scores, strengths, and red flags.
Report Generation: Create detailed recruitment reports in PDF and CSV formats.
Interview Management: Conduct interactive interviews with automated follow-up questions and summaries.
Email Automation: Send congratulatory, feedback, and interview schedule emails.
Job Description Generation: Generate professional job descriptions from key terms.
Vector Database: Use FAISS for efficient resume storage and similarity search.
Requirements
Python 3.8 or higher
Required Python packages (install via requirements.txt):
flask
pandas
python-docx
pdfplumber
groq
sentence-transformers
numpy
faiss-cpu
reportlab
python-dotenv
sqlite3 (included with Python)
Installation
1. Clone the Repository
bash

Collapse

Wrap

Run

git clone [https://github.com/yourusername/recruitment-agent.git](https://github.com/Tariq3654467/Flask_Recruitpro)
cd recruitment-agent
2. Install Dependencies
Create and activate a virtual environment (optional but recommended):


python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install the required packages:



Copy
pip install -r requirements.txt
3. Set Up Environment Variables
Create a .env file in the project root with the following variables:

Copy
GROQ_API_KEY=your_groq_api_key_here
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
BASE_URL=http://your_domain_or_localhost:5000
Replace your_groq_api_key_here with a valid Groq API key.
4. Run the Application
Start the Flask server:

bash

Collapse

Wrap

Run

Copy
python app.py
The application will be available at http://localhost:5000.

Usage
Uploading Resumes
Navigate to http://localhost:5000.
Enter a job description and upload one or more PDF/DOCX resumes.
Submit to process resumes and receive candidate data with scores and analysis.
Generating Reports
Use the /api/generate_report endpoint (POST) with candidates and job_description to get a Markdown report.
Download as PDF via /api/download_pdf (POST).
Conducting Interviews
Start an interview with /api/start_interview (POST) by providing candidate_email, candidate_name, and job_description.
Use /api/get_interview_state (GET) with interview_id to view the conversation.
Submit candidate responses with /api/submit_candidate_response (POST) and agent questions with /api/submit_agent_question (POST).
End the interview with /api/end_interview (POST) to generate and send a summary.
Email Automation
Send congratulatory emails with /api/send_congratulatory_email (POST) using selected_candidates.
Send feedback emails with /api/send_feedback_email (POST) using unselected_candidates.
Schedule interviews with /api/send_interview_schedule (POST) using selected_candidates and schedule.
Generating Job Descriptions
Use /api/generate_job_description (POST) with a list of key_terms to generate a job description.
API Endpoints

Endpoint	Method	Description	Parameters
/api/process_resumes	POST	Process uploaded resumes	job_description, resumes
/api/generate_report	POST	Generate a recruitment report	candidates, job_description
/api/ask_question	POST	Answer a recruitment question	query, candidates, job_description
/api/export_csv	POST	Export candidate data as CSV	candidates
/api/download_pdf	POST	Download report as PDF	candidates, job_description
/api/send_congratulatory_email	POST	Send congratulatory emails	selected_candidates
/api/send_feedback_email	POST	Send feedback emails	unselected_candidates
/api/send_interview_schedule	POST	Send interview schedule emails	selected_candidates, schedule
/api/generate_job_description	POST	Generate a job description	key_terms
/api/start_interview	POST	Start an interview session	candidate_email, candidate_name, job_description
/api/get_interview_state	GET	Get current interview state	interview_id
/api/submit_agent_question	POST	Submit an agent question	interview_id, question
/api/submit_candidate_response	POST	Submit a candidate response	interview_id, response
/api/end_interview	POST	End an interview and generate summary	interview_id
Contributing
Fork the repository.
Create a new branch (git checkout -b feature-branch).
Make changes and commit (git commit -m "Add new feature").
Push to the branch (git push origin feature-branch).
Open a Pull Request with a clear description of changes.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For support or questions, please open an issue on the GitHub repository or contact the maintainers at thinkrecruit1@gmail.com.com.
