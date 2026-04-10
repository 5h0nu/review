import os
import json
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Allows your frontend to communicate with this API
CORS(app)

# API CONFIG - Set NVIDIA_API_KEY in Vercel Dashboard -> Settings -> Environment Variables
API_KEY = os.environ.get("NVIDIA_API_KEY")
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
    if not API_KEY:
        return {"error": "API Key missing in environment variables"}
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
        payload = {
            "model": model, 
            "messages": [{"role": "user", "content": f"{AUDIT_LOGIC}\n\nDATA:\n{content}"}], 
            "temperature": 0.1
        }
        r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=25)
        response_data = r.json()
        
        raw_content = response_data['choices'][0]['message']['content']
        # Remove Markdown formatting if the model adds it
        clean_json = raw_content.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except Exception as e:
        print(f"Error calling {model}: {str(e)}")
        return None

@app.route('/')
def home():
    return "<h1>ReviewIntel Node: Online</h1><p>Vercel Deployment Active.</p>"

@app.route('/analyze-dual', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS': 
        return '', 200
        
    try:
        # Check if input is a file (CSV) or raw text
        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
            content = df.head(10).to_string() # Analyze top 10 rows
        else:
            content = request.form.get('text', 'No data provided')

        with ThreadPoolExecutor(max_workers=2) as ex:
            # Parallel execution for speed
            future_gemma = ex.submit(call_api, "google/gemma-2-9b-it", content)
            future_llama = ex.submit(call_api, "meta/llama-3.1-70b-instruct", content)
            
            r1 = future_gemma.result()
            r2 = future_llama.result()

        # Return Gemma result if available, otherwise Llama
        result = r1 if r1 else r2
        if not result:
            return jsonify({"error": "Both AI models failed to respond"}), 502
            
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Vercel doesn't use app.run(), but we keep it for local testing
if __name__ == '__main__':
    app.run(debug=True)
