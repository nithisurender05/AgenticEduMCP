import json
import os
import asyncio
import re
from google import genai
from google.genai.errors import ServerError, ClientError

api_key = os.environ["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

MODEL_NAME = "gemini-2.5-flash"


async def safe_generate(**kwargs):
    for i in range(5):
        try:
            return await asyncio.to_thread(
                client.models.generate_content,
                **kwargs
            )
        except (ServerError, ClientError) as e:
            print(f"Retry {i+1}/5 due to error: {e}")
            await asyncio.sleep(2 ** i)

    raise RuntimeError("LLM judge failed after retries")


def extract_json(text):
    if not text:
        raise ValueError("Empty LLM response")

    # remove markdown 
    text = text.strip()
    text = re.sub(r"```json|```", "", text)

    # extract first JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in response:\n{text}")

    return json.loads(match.group(0))


with open("results.json") as f:
    results = json.load(f)

with open("mock_canvas_data.json") as f:
    student_db = json.load(f)



def get_student(student_id):
    for s in student_db["student_profiles"]:
        if str(s["id"]) == str(student_id):
            return s
    return None


# heuristic scoring
def heuristic_score(text, student):
    text_l = (text or "").lower()

    if student is None:
        return {
            "grounding": 0,
            "personalization": 0,
            "specificity": 0,
            "actionability": 0
        }

    grades = student["courses"][0]["grades"]
    assignments = [g["assignment"].lower() for g in grades]

    # grounding
    grounding_hits = sum(1 for a in assignments if a in text_l)
    numeric_hits = sum(str(g["score"]) in text_l for g in grades)
    grounding = min((grounding_hits + numeric_hits) / (2 * len(grades)), 1.0)

    # personalization
    personalization = 1.0 if student["name"].lower() in text_l else 0.0

    # specificity
    keywords = [
        "quiz", "midterm", "partial fractions",
        "integration", "differential", "trig"
    ]
    specificity = min(sum(k in text_l for k in keywords) / len(keywords), 1.0)

    # actionability
    action_words = ["practice", "step", "first", "then", "try", "solve"]
    actionability = min(sum(w in text_l for w in action_words) / len(action_words), 1.0)

    return {
        "grounding": grounding,
        "personalization": personalization,
        "specificity": specificity,
        "actionability": actionability
    }


# llm judge prompt
def build_judge_prompt(student_id, baseline, contextual, full_contextual, agentic):
    return f"""
You are an expert evaluator of AI tutoring systems.

Student ID: {student_id}

Evaluate the following responses:

--- BASELINE ---
{baseline}

--- CONTEXTUAL ---
{contextual}

--- FULL CONTEXT UPPER BOUND ---
{full_contextual}

--- AGENTIC (MCP TOOL-USE SYSTEM) ---
{agentic}

Score each response from 0 to 2:

Groundedness:
0 = hallucination / ignores data
1 = partial grounding
2 = fully grounded in student data

Personalization:
0 = generic
1 = partial reference
2 = fully tailored

Specificity:
0 = vague
1 = some concepts
2 = precise weaknesses

Actionability:
0 = passive
1 = general suggestions
2 = step-by-step plan

IMPORTANT:
Return ONLY valid JSON.
No markdown. No explanation.

Format:
{{
  "baseline": {{}},
  "contextual": {{}},
  "full_contextual": {{}},
  "agentic": {{}},
  "winner": "",
  "justification": ""
}}
"""


# main loop
async def main():
    outputs = []

    for r in results:
        student = get_student(r["student_id"])

        print(f"Processing student {r['student_id']}...")

        heuristic = {
            "baseline": heuristic_score(r.get("baseline", ""), student),
            "contextual": heuristic_score(r.get("contextual", ""), student),
            "full_contextual": heuristic_score(r.get("full_contextual", ""), student),
            "agentic": heuristic_score(r.get("agentic", ""), student),
        }

        prompt = build_judge_prompt(
            r["student_id"],
            r.get("baseline", ""),
            r.get("contextual", ""),
            r.get("full_contextual", ""),
            r.get("agentic", "")
        )

        response = await safe_generate(
            model=MODEL_NAME,
            contents=prompt,
            config={
                "response_mime_type": "application/json"
            }
        )

        try:
            llm_judge = extract_json(response.text)
        except Exception as e:
            print("Failed to parse LLM response.")
            print("Error:", e)
            print("RAW OUTPUT:\n", response.text)
            continue

        outputs.append({
            "student_id": r["student_id"],
            "heuristic": heuristic,
            "llm_judge": llm_judge
        })

    with open("evaluation_results.json", "w") as f:
        json.dump(outputs, f, indent=2)

    print("Saved evaluation_results.json")



if __name__ == "__main__":
    asyncio.run(main())