import os
import json
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
CORS(app)

API_KEY = "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"
INVOKE_URL = "https://integrate.api.nvidia.com/v1/chat/completions"

# THE MASTER PROMPT (Now includes Full Review Data for Popups)
MASTER_LOGIC = """
Return ONLY JSON:
{
  "total_reviews": 8,
  "categories": ["Electronics", "Kitchen", "Home Decor"],
  "languages_detail": "English (70%), Kannada (30%). Multilingual NLP Active ✅",
  "health_analysis": "Current score of 65% is driven by high logistics failure in Kitchen. Churn risk is moderate.",
  "all_reviews": [
    {"id": 1, "cat": "Electronics", "txt": "Battery is amazing, last 2 days.", "sent": "Positive"},
    {"id": 2, "cat": "Electronics", "txt": "Quality thumba chennagide (Quality is very good).", "sent": "Positive"},
    {"id": 4, "cat": "Electronics", "txt": "Great job on quality control! Failed in 1 hour.", "sent": "Negative/Sarcastic"},
    {"id": 5, "cat": "Kitchen", "txt": "Item broken on arrival.", "sent": "Negative"},
    {"id": 6, "cat": "Kitchen", "txt": "Box was torn and item damaged.", "sent": "Negative"}
  ],
  "category_summaries": {
    "Electronics": {"pos": 3, "neg": 1, "mixed": 1, "issues": "Product failure", "strengths": "Battery/Support"},
    "Kitchen": {"pos": 0, "neg": 3, "mixed": 0, "issues": "Packaging failure", "strengths": "None"},
    "Home Decor": {"pos": 0, "neg": 0, "mixed": 1, "issues": "Delivery damage", "strengths": "Fast delivery"}
  },
  "key_problems": {
    "packaging": {"title": "Systemic Packaging Failure", "detail": "Critical failure in Kitchen transit. 100% of complaints involve torn boxes.", "reviews": [5, 6, 7]},
    "quality": {"title": "Product Quality (Electronics)", "detail": "Internal circuit failures reported in Review #4. Potential batch defect.", "reviews": [4]},
    "sarcasm": {"title": "Sarcasm Detected", "detail": "Linguistic analysis found sarcasm in Review #4 where 'Great job' actually meant failure.", "reviews": [4]}
  },
  "recommendations": ["Double-layer boxes for Kitchen", "Pre-shipment testing for Electronics"],
  "health_score": 65
}
"""

def call_api(model, content):
    try:
        headers = {"Authorization": f"Bearer {API_KEY}", "Accept": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": f"{MASTER_LOGIC}\n\nDATA:\n{content}"}], "temperature": 0.1}
        r = requests.post(INVOKE_URL, headers=headers, json=payload, timeout=15)
        return json.loads(r.json()['choices'][0]['message']['content'].replace('```json', '').replace('```', '').strip())
    except: return None

@app.route('/analyze-dual', methods=['POST', 'OPTIONS'])
def analyze():
    # TEST MOCK DATA - Use this to see if your UI displays correctly
    test_data = {
        "total_reviews": 8,
        "categories": ["A", "B"],
        "health_score": 90,
        "all_reviews": [],
        "category_summaries": {},
        "key_problems": {"packaging": {"title":"Test", "detail":"Test", "reviews":[]}}
    }
    return jsonify({"gemma": test_data})
if __name__ == '__main__':
    app.run(debug=False, port=8000, host='127.0.0.1', threaded=True)
