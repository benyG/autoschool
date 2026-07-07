import os
from openai import OpenAI
from src.vector_store import search

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

THEMES = [
    "Panneaux de signalisation",
    "Règles de priorité et intersections",
    "Limites de vitesse",
    "Alcool et drogues au volant",
    "Distances de sécurité et freinage",
    "Feux de circulation",
    "Stationnement et arrêt",
    "Conduite hivernale",
    "Partage de la route (piétons, cyclistes, motos)",
    "Manœuvres : dépassement, virage, marche arrière",
    "Points d'inaptitude et infractions",
    "Permis de conduire et étapes au Québec",
]


def generate_summary(theme: str, openai_api_key: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    context_chunks = search(theme, openai_api_key, n_results=6)
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un formateur expert en code de la route au Québec. "
                    "Tu crées des fiches de révision claires, concises et mémorables "
                    "pour aider les candidats à réussir leur examen SAAQ. "
                    "Utilise des listes à puces, des chiffres clés en gras, et un langage simple."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Crée une fiche de révision complète sur le thème : **{theme}**\n\n"
                    f"Voici les extraits du manuel officiel du Québec :\n\n{context}\n\n"
                    "La fiche doit inclure :\n"
                    "- Les règles essentielles à retenir\n"
                    "- Les chiffres importants (vitesses, distances, durées)\n"
                    "- Les erreurs fréquentes à éviter\n"
                    "- Un conseil mnémotechnique si possible"
                ),
            },
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content
