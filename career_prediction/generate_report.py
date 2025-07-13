import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues

import json, datetime, uuid, os
from jinja2 import Template
from xhtml2pdf import pisa
import matplotlib.pyplot as plt

SKILL_FILE = "career_skills.json"
OUTPUT_DIR = "reports"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_chart_image(top3, file_id):
    labels = [label for label, _ in top3]
    scores = [score for _, score in top3]

    fig, ax = plt.subplots(figsize=(6, 3))
    bars = ax.barh(labels, scores, color='#1e90ff')
    ax.set_xlim(0, 100)
    ax.set_xlabel('Confidence (%)')
    ax.set_title('Career Confidence Levels')

    for bar in bars:
        width = bar.get_width()
        ax.text(width - 5, bar.get_y() + bar.get_height()/2, f'{width:.1f}%', va='center', ha='right', color='white')

    chart_path = os.path.join(OUTPUT_DIR, f"chart_{file_id}.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    return chart_path

def build_html(user, top3, recommendations, chart_path):
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    template = Template("""<!DOCTYPE html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Career Report</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f8f9fa;
      color: #2c3e50;
      font-size: 14px;
    }
    .container {
      background: white;
      max-width: 800px;
      margin: auto;
      padding: 30px;
      border-radius: 10px;
      border: 1px solid #ddd;
    }
    .header {
      display: flex;
      justify-content: center;
      align-items: center;
      margin-bottom: 10px;
      gap: 20px;
      
    }
    .header img {
      height: 60px;
    }
    .banner {
      max-height: 50px;
    }
    h1 {
      text-align: center;
      color: #1e90ff;
      margin-bottom: 10px;
    }
    .date {
      text-align: center;
      font-size: 12px;
      color: #888;
      margin-bottom: 25px;
    }
    .section {
      margin-bottom: 25px;
    }
    .section h2 {
      background-color: #1e90ff;
      color: white;
      padding: 8px 12px;
      font-size: 16px;
      border-radius: 4px;
      margin-bottom: 12px;
    }
    .highlight {
      font-size: 16px;
      font-weight: bold;
      color: #2c3e50;
      margin: 10px 0;
    }
    ul {
      margin-top: 6px;
      margin-left: 18px;
      padding-left: 0;
    }
    li {
      margin-bottom: 5px;
    }
    .tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}

.tag {
  background: linear-gradient(to right, #dceeff, #f0f8ff);
  color: #003366;
  padding: 6px 14px;
  border-radius: 20px;
  font-size: 13px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  font-weight: 500;
}

    .course-link {
      font-weight: bold;
      color: #1e90ff;
      font-size: 13px;
    }
    footer {
      text-align: center;
      font-size: 11px;
      color: #999;
      margin-top: 40px;
      border-top: 1px solid #ddd;
      padding-top: 10px;
    }
  </style>
</head>
<body>
<div class="header" style="display: table; margin: 0 auto; text-align: center;">
  <table style="margin: auto;">
    <tr>
      <td><img src="static/images/CubeAI_Logo.png" alt="Left Logo" style="height: 60px;"></td>
      <td style="padding: 0 20px;"><img src="static/images/CubeAiSolutiion.png" alt="Company Tagline" style="height: 80px;"></td>
      <td><img src="static/images/CubeAI_Logo.png" alt="Right Logo" style="height: 60px;"></td>
    </tr>
  </table>
</div>


  <h1>🚀 Career Report – {{ user.name }}</h1>
  <div class="date">📅 Generated on {{ now }}</div>

  <div class='section'>
    <h2>🎯 Top Career Recommendation</h2>
    <div class="highlight"> 🔹  {{   top3[0][0] }}</div>
  </div>

  <div class='section'>
    <h2>📈 Career Confidence Chart</h2>
    <img src="{{ chart_path }}" alt="Career Chart" style="width:600px; height:auto; border: 1px solid #ccc; border-radius: 8px;">
  </div>

  <div class='section'>
    <h2>👤 User Profile</h2>
  <ul>
  <li><strong>Email:</strong> {{ user.email }}</li>
  <li><strong>Stream:</strong> {{ user.stream }}</li>
  <li><strong>CGPA:</strong> {{ user.cgpa }}</li>
  <li><strong>Certifications:</strong> {{ user.certifications }}</li>
  <li><strong>Internships:</strong> {{ user.internships }}</li>
   <li><strong>Skills:</strong> <div class="tags">
  {% for skill in user.skills.split(",") %}
    <span class="tag">🔹 {{ skill.strip() }}</span>
  {% endfor %}
</div></li>
</ul>

   


  <div class='section'>
    <h2>🛠️ Skills to Master</h2>
    <ul>
      {% for sk in recommendations.skills %}
        <li>{{ sk }}</li>
      {% endfor %}
    </ul>
  </div>

  <div class='section'>
    <h2>📚 Recommended Courses</h2>
    <ul>
      {% for course in recommendations.courses %}
        <li>
          <span class="course-link">{{ course.title }}</span><br/>
          <small>{{ course.url }}</small>
        </li>
      {% endfor %}
    </ul>
  </div>

  <footer>
    Generated by Career Prediction System
  </footer>
</div>
</body>
</html>
""")
    return template.render(user=user, top3=top3, recommendations=recommendations, now=now, chart_path=chart_path)

def generate_report(user_data, top3):
    with open(SKILL_FILE) as f:
        skills_db = json.load(f)
    career_key = top3[0][0]
    career_data = skills_db.get(career_key, {"skills": [], "courses": []})

    file_id = str(uuid.uuid4())[:8]
    chart_path = create_chart_image(top3, file_id)

    html_content = build_html(user_data, top3, career_data, chart_path)
    pdf_path = os.path.join(OUTPUT_DIR, f"career_report_{file_id}.pdf")
    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(html_content, dest=f)
    return pdf_path
