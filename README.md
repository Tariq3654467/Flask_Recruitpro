Recruitment Interview Automation Tool
This is a Flask-based web application designed to automate the recruitment interview process. It allows for resume processing, candidate interviews via a chat interface, automated question generation, interview summaries, and email notifications. The tool leverages AI models (via the Groq API) for natural language processing and integrates with a SQLite database for storing interview data.
Features

Process resumes (PDF/DOCX) and evaluate candidate fit against a job description.
Generate automated follow-up questions during interviews.
Conduct interviews via a web-based chat interface.
Automatically end interviews after 24 hours of inactivity and send summaries.
Send interview schedules, congratulatory emails, feedback emails, and interview summaries via SMTP.
Export candidate rankings to CSV.
Generate job descriptions and recruitment reports.

Prerequisites

Python 3.8+
Required Python packages (see requirements.txt).
Groq API key (set as GROQ_API_KEY in .env).
SMTP server credentials (set as SENDER_EMAIL and SENDER_PASSWORD in .env).
Hiring team email (set as HIRING_TEAM_EMAIL in .env).

Installation
1. Clone the Repository
git clone https://github.com/yourusername/recruitment-interview-tool.git
cd recruitment-interview-tool

2. Install Dependencies
Create a virtual environment and install the required packages:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

3. Set Up Environment Variables
Create a .env file in the project root with the following content:
GROQ_API_KEY=your_groq_api_key
SENDER_EMAIL=your_email@example.com
SENDER_PASSWORD=your_email_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
HIRING_TEAM_EMAIL=hiring_team@example.com
BASE_URL=http://localhost:5000


Replace your_groq_api_key, your_email@example.com, your_email_password, and hiring_team@example.com with your actual values.
Ensure your SMTP server (e.g., Gmail) allows less secure apps or uses an app-specific password if 2FA is enabled.

4. Initialize the Database
The application creates a SQLite database (interviews.db) automatically on startup. Ensure the static folder exists with index.html and candidate_interview.html.
5. Run the Application
Start the Flask server:
python app.py

Access the application at http://localhost:5000.
Usage
1. Upload Resumes

Navigate to the root URL (/) and upload resumes (PDF/DOCX) along with a job description.
The system processes resumes and returns candidate rankings with fit scores.

2. Schedule Interviews

Use the /api/send_interview_schedule endpoint to send interview invitations to selected candidates.
Candidates receive an email with a link to candidate_interview.

3. Conduct Interviews

Candidates access the interview via the provided link and respond to questions.
The agent generates follow-up questions automatically.
Interviews end after 24 hours of inactivity, and summaries are sent to the hiring team.

4. End Interview Manually

Use the /api/end_interview endpoint with an interview_id to manually end an interview and generate a summary.

5. Additional Features

Generate job descriptions via /api/generate_job_description.
Export candidate data to CSV via /api/export_csv.
Send congratulatory or feedback emails via respective endpoints.

API Endpoints

/api/process_resumes: Process resumes and evaluate candidates.
/api/send_interview_schedule: Send interview schedule emails.
/api/start_interview: Start a new interview session.
/api/get_interview_state: Get the current state of an interview.
/api/submit_candidate_response: Submit a candidate's response.
/api/end_interview: End an interview and send a summary.
/api/generate_report: Generate a recruitment report.
Others as listed in app.py.

File Structure
recruitment-interview-tool/
├── static/
│   ├── index.html
│   └── candidate_interview.html
├── app.py
├── requirements.txt
├── .env
├── Dockerfile
└── README.md

