import json

from openai import OpenAI

from src.vector_store import search
from src.models import get_chat_model

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

_MONITEUR = (
    "Tu es un moniteur d'auto-école québécois chevronné, du genre que tout le monde "
    "recommande parce qu'il rend les choses limpides. Tu prépares un élève à l'examen "
    "théorique de la SAAQ. Il n'est pas là pour survoler : il est là pour le réussir.\n\n"
    "Ta façon de parler :\n"
    "- Tutoie l'élève, parle-lui directement (« regarde », « imagine que tu arrives à »).\n"
    "- Écris comme tu parles : phrases courtes, ton chaleureux, zéro jargon administratif.\n"
    "- Explique toujours le POURQUOI avant le QUOI. Une règle comprise ne s'oublie plus.\n"
    "- Ancre chaque notion dans une situation de conduite réelle au Québec.\n"
    "- Bannis les formules robotiques : pas de « Il est important de noter que », "
    "pas de « En conclusion », pas d'énumération sèche de règlements."
)


def _analyse_points(theme: str, context: str, client: OpenAI) -> list[dict]:
    """Premier passage : trie les notions du thème par criticité d'examen.

    C'est ce tri qui empêche le second passage de compresser une règle de priorité
    autant qu'un détail anecdotique.
    """
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es examinateur à la SAAQ. Tu sais exactement ce qui tombe à l'examen "
                    "théorique, ce qui fait échouer les candidats, et ce qui n'est que du "
                    "contexte. Réponds UNIQUEMENT en JSON valide."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Voici des extraits du manuel officiel du Québec sur le thème "
                    f"« {theme} » :\n\n{context}\n\n"
                    "Identifie les notions distinctes présentes dans ces extraits et classe "
                    "chacune selon son enjeu à l'examen théorique :\n"
                    "- \"critique\" : tombe souvent, et se tromper coûte des points ou fait "
                    "échouer. Typiquement les règles de priorité, les seuils chiffrés, les "
                    "obligations légales.\n"
                    "- \"important\" : peut tomber, mérite d'être compris.\n"
                    "- \"secondaire\" : contexte utile, mais pas de quoi faire échouer.\n\n"
                    "Pour chaque notion, indique aussi la confusion classique : avec quoi les "
                    "candidats la mélangent-ils, et pourquoi.\n\n"
                    "JSON exact :\n"
                    '{"points": [{"titre": "...", "criticite": "critique|important|secondaire", '
                    '"enjeu_examen": "...", "confusion_frequente": "...", '
                    '"chiffres_cles": ["..."]}]}\n\n'
                    "N'invente rien : ne retiens que ce qui figure réellement dans les extraits."
                ),
            },
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    try:
        return json.loads(response.choices[0].message.content).get("points", [])
    except (json.JSONDecodeError, TypeError):
        return []


def _format_triage(points: list[dict]) -> str:
    """Met le tri à plat pour le prompt de rédaction."""
    if not points:
        return ""

    lines = []
    for niveau in ("critique", "important", "secondaire"):
        groupe = [p for p in points if p.get("criticite") == niveau]
        if not groupe:
            continue
        lines.append(f"\nNotions de niveau {niveau.upper()} :")
        for p in groupe:
            lines.append(f"- {p.get('titre', '?')}")
            if p.get("enjeu_examen"):
                lines.append(f"  · enjeu à l'examen : {p['enjeu_examen']}")
            if p.get("confusion_frequente"):
                lines.append(f"  · confusion fréquente : {p['confusion_frequente']}")
            if p.get("chiffres_cles"):
                lines.append(f"  · chiffres : {', '.join(p['chiffres_cles'])}")
    return "\n".join(lines)


def generate_summary(theme: str, openai_api_key: str) -> tuple[str, list[dict]]:
    """Rédige la fiche du thème avec une profondeur graduée selon l'enjeu d'examen.

    Retourne (texte de la fiche, points triés).
    """
    client = OpenAI(api_key=openai_api_key)
    context = "\n\n".join(search(theme, openai_api_key, n_results=8))

    points = _analyse_points(theme, context, client)
    triage = _format_triage(points)

    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {"role": "system", "content": _MONITEUR},
            {
                "role": "user",
                "content": (
                    f"Prépare-moi le thème : **{theme}**\n\n"
                    f"Extraits du manuel officiel du Québec :\n\n{context}\n"
                    f"{'---' + chr(10) + 'Un examinateur a trié les notions par enjeu :' + chr(10) + triage if triage else ''}\n\n"
                    "RÈGLE DE PROFONDEUR — c'est le point le plus important de ta rédaction.\n"
                    "Tu n'écris pas un résumé uniforme. Tu doses selon l'enjeu :\n\n"
                    "▸ Pour CHAQUE notion critique, déroule le traitement complet, sans "
                    "jamais la compresser pour gagner de la place :\n"
                    "   1. ce que l'examen attend précisément sur ce point ;\n"
                    "   2. la règle exacte, avec les chiffres repris **mot pour mot** du "
                    "manuel, en gras ;\n"
                    "   3. la logique derrière — pourquoi cette règle existe ;\n"
                    "   4. une situation concrète déroulée du début à la fin, où l'on te voit "
                    "appliquer la règle au volant ;\n"
                    "   5. le piège : avec quoi on la confond, et ce qui permet de trancher "
                    "entre les deux ;\n"
                    "   6. une question de vérification, suivie de sa réponse — pour que je "
                    "teste ma compréhension au lieu de relire passivement.\n\n"
                    "▸ Pour les notions importantes : la règle, sa logique et son piège, en un "
                    "paragraphe dense mais complet.\n\n"
                    "▸ Pour les notions secondaires : une phrase, regroupées en fin de fiche "
                    "sous « Bon à savoir ». Ne leur donne pas plus de place que ça.\n\n"
                    "Ouvre la fiche par une phrase qui me dit ce que ce thème peut me coûter "
                    "à l'examen, et termine par un moyen mnémotechnique ou une image mentale "
                    "qui tienne dans la tête le jour J.\n\n"
                    "La longueur n'est pas un critère : une notion critique traitée en trois "
                    "lignes est un échec, même si la fiche est jolie. Ce qui compte, c'est "
                    "qu'en la lisant ou en l'écoutant je gagne en compréhension et en "
                    "assurance.\n\n"
                    "Reste strictement fidèle au manuel : n'invente aucune règle ni aucun "
                    "chiffre. Si un point n'est pas couvert par les extraits, dis-le "
                    "franchement plutôt que de combler le vide."
                ),
            },
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content, points
