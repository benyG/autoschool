import json
from openai import OpenAI

from src.models import get_chat_model
from src.vector_store import search
from src.summaries import THEMES
import random


def generate_exam(openai_api_key: str, n_questions: int = 20) -> list[dict]:
    client = OpenAI(api_key=openai_api_key)
    questions = []
    themes_sample = random.sample(THEMES, min(len(THEMES), n_questions))

    for theme in themes_sample:
        context_chunks = search(theme, openai_api_key, n_results=3)
        context = "\n\n".join(context_chunks)

        response = client.chat.completions.create(
            model=get_chat_model(),
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Tu es un examinateur de la SAAQ au Québec. "
                        "Tu génères des questions d'examen réalistes à choix multiples. "
                        "Réponds UNIQUEMENT en JSON valide."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Génère 1 question QCM sur le thème : {theme}\n\n"
                        f"Extraits du manuel :\n{context}\n\n"
                        "JSON exact :\n"
                        '{"question": "...", "choices": ["A. ...", "B. ...", "C. ...", "D. ..."], '
                        '"correct_index": 0, "explanation": "..."}'
                    ),
                },
            ],
            temperature=0.5,
            response_format={"type": "json_object"},
        )

        try:
            q = json.loads(response.choices[0].message.content)
            q["theme"] = theme
            questions.append(q)
        except Exception:
            continue

    return questions[:n_questions]


def evaluate_exam(questions: list[dict], answers: list[int]) -> dict:
    results = []
    score = 0
    for i, (q, user_ans) in enumerate(zip(questions, answers)):
        correct = q.get("correct_index", 0)
        is_correct = user_ans == correct
        if is_correct:
            score += 1
        results.append({
            "question": q["question"],
            "choices": q["choices"],
            "user_answer": user_ans,
            "correct_answer": correct,
            "is_correct": is_correct,
            "explanation": q.get("explanation", ""),
            "theme": q.get("theme", ""),
        })

    passing = score >= round(len(questions) * 0.8)  # 80% pour réussir
    return {
        "score": score,
        "total": len(questions),
        "percentage": round(score / len(questions) * 100) if questions else 0,
        "passed": passing,
        "results": results,
    }
