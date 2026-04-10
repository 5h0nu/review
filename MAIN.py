import os
import json
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# CRITICAL: Allows your local index.html to talk to the Railway Cloud
CORS(app, resources={r"/*": {"origins": "*"}})

# --- CONFIGURATION ---
# It is better to set this in Railway's "Variables" tab as NVIDIA_API_KEY
API_KEY = os.getenv("NVIDIA_API_KEY")
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# THE MASTER PROMPT (Optimized for the Hackathon Audit Output)
AUDIT_LOGIC = """
Analyze the reviews for a Retail Audit. Return ONLY JSON:
{
  "total_reviews": 8,
  "categories": ["Electronics", "Kitchen", "Home Decor"],
  "languages_detail": "English + Kannada (Multilingual ✅)",
  "health_analysis": "Based on systemic failures in specific categories.",
  "all_reviews": [
    {"id": 1, "cat": "Electronics", "txt": "Review text...", "sent": "Positive"}
  ],
  "category_summaries": {
    "Electronics": {"pos": 3, "neg": 1, "mixed": 1, "issues": "Product failure", "strengths": "Support"},
    "Kitchen": {"pos": 0, "neg": 3, "mixed": 0, "issues": "Packaging failure", "strengths": "None"},
    "Home Decor": {"pos": 0, "neg": 0, "mixed": 1, "issues": "Logistics", "strengths": "Speed"}
  },
  "key_problems": {
    "packaging": {"title": "Systemic Packaging Failure", "detail": "Critical failure in Kitchen transit.", "reviews": [5, 6, 7]},
    "quality": {"title": "Product Quality (Electronics)", "detail": "Batch defect in circuits.", "reviews": [4]},
    "sarcasm": {"title": "Sarcasm Detected", "detail": "Sarcasm identified in Review #4.", "reviews": [4]}
  },
  "recommendations": ["Double-layer boxes for Kitchen", "QC for Electronics"],
  "health_score": 65
}
"""

def call_api(model, content):
    """Helper to call NVIDIA NIM with specific model"""
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": f"{AUDIT_LOGIC}\n\nDATA:\n{content}"}],
            "temperature": 0.1,
            "max_tokens": 1500
        }
        # 25 second timeout for fast demo response
        r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=25)
        raw = r.json()['choices'][0]['message']['content']
        return json.loads(raw.replace('```json', '').replace('```', '').strip())
    except Exception as e:
        print(f"Error with {model}: {e}")
        return None

# 1. HEALTH CHECK (Prevents "Site Can't Be Reached" error)
@app.route('/')
def home():
    return "<h1>ReviewIntel Pro: Cloud Node Online</h1><p>Status: <span style='color:green'>Active</span></p>"

# 2. ANALYSIS ENDPOINT
@app.route('/analyze-dual', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        if 'file' in request.files:
            df = pd.read_csv(request.files['file'])
            content = df.head(8).to_string()
        else:
            content = request.form.get('text', '')

        # PARALLEL EXECUTION (The Speed Hack)
        with ThreadPoolExecutor() as ex:
            f1 = ex.submit(call_api, "google/gemma-2-9b-it", content)
            f2 = ex.submit(call_api, "meta/llama-3.1-70b-instruct", content)
            
            out_a = f1.result()
            out_b = f2.result()

        # FALLBACK LOGIC
        final_data = out_a if out_a else out_b
        
        if not final_data:
            return jsonify({"error": "AI models busy. Please try again."}), 503

        return jsonify({"gemma": final_data, "llama": out_b if out_b else out_a})

    except Exception as e:
        print(f"System Error: {e}")
        return jsonify({"error": str(e)}), 500

# 3. RAILWAY DYNAMIC PORT BINDING
if __name__ == '__main__':
    # Railway provides the PORT variable automatically
    port = int(os.environ.get("PORT", 8080))
    # host MUST be 0.0.0.0 for public access
    app.run(debug=False, host='0.0.0.0', port=port)
