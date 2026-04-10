import os
import json
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# This must be set to allow your local HTML to talk to Railway
CORS(app, resources={r"/*": {"origins": "*"}})

# API CONFIG
API_KEY = os.environ["NVIDIA_API_KEY"]
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

AUDIT_LOGIC = """
Return ONLY JSON:
{
  "total_reviews": 8,
  "categories": ["Electronics", "Kitchen", "Home Decor"],
  "languages_detail": "English + Kannada (Multilingual ✅)",
  "health_analysis": "Based on systemic packaging failures.",
  "all_reviews": [{"id": 1, "cat": "Electronics", "txt": "Good!", "sent": "Positive"}],
  "category_summaries": {
    "Electronics": {"pos": 3, "neg": 1, "mixed": 1, "issues": "Product failure", "strengths": "Support"},
    "Kitchen": {"pos": 0, "neg": 3, "mixed": 0, "issues": "Packaging failure", "strengths": "None"},
    "Home Decor": {"pos": 0, "neg": 0, "mixed": 1, "issues": "Logistics", "strengths": "Speed"}
  },
  "key_problems": {
    "packaging": {"title": "Systemic Packaging Failure", "detail": "Critical Kitchen transit fail.", "reviews": [1]},
    "quality": {"title": "Product Quality", "detail": "Batch defect.", "reviews": [1]},
    "sarcasm": {"title": "Sarcasm Detected", "detail": "Linguistic mismatch.", "reviews": [1]}
  },
  "health_score": 65
}
"""

def call_api(model, content):
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": f"{AUDIT_LOGIC}\n\nDATA:\n{content}"}], "temperature": 0.1}
        r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=20)
        raw = r.json()['choices'][0]['message']['content']
        return json.loads(raw.replace('```json', '').replace('```', '').strip())
    except: return None

@app.route('/')
def home():
    return "<h1>ReviewIntel Node: Online</h1>"

@app.route('/analyze-dual', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS': return '', 200
    try:
        content = pd.read_csv(request.files['file']).head(8).to_string() if 'file' in request.files else request.form.get('text', '')
        with ThreadPoolExecutor() as ex:
            r1 = ex.submit(call_api, "google/gemma-2-9b-it", content).result()
            r2 = ex.submit(call_api, "meta/llama-3.1-70b-instruct", content).result()
        return jsonify({"gemma": r1 if r1 else r2})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8000))
    app.run(host='0.0.0.0', port=port)
