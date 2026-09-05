"""Recoupement du contenu généré avec le manuel officiel.

Deux étages, du moins cher au plus cher :

1. Contrôle numérique déterministe — aucun appel API. Chaque valeur chiffrée
   accompagnée d'une unité (80 km/h, 0,08, 3 secondes) doit se retrouver dans
   les extraits du manuel. C'est là que se logent les hallucinations les plus
   coûteuses pour un candidat, et c'est vérifiable exactement.

2. Contrôle des affirmations normatives par le modèle — chaque assertion est
   recoupée avec une recherche fraîche dans le manuel, pour ne pas la déclarer
   absente simplement parce qu'elle sortait de la fenêtre de récupération
   initiale.
"""
import json
import re
import unicodedata

from openai import OpenAI

from src.models import get_chat_model
from src.vector_store import search

# ── Étage 1 : chiffres ────────────────────────────────────────────────────────

# Unités qui rendent un nombre vérifiable. Un « 2 » nu dans une phrase n'est pas
# une affirmation factuelle ; « 2 secondes » en est une.
_UNITS = (
    r"km/h|km|m(?:ètres?)?|cm|mm|"
    r"secondes?|sec|s|minutes?|min|heures?|h|jours?|semaines?|mois|ans?|années?|"
    r"\$|%|pour ?cent|points?|g/l|mg|ml|litres?|l|"
    r"passagers?|occupants?|véhicules?|roues?"
)
_NUMBER = r"\d+(?:[  ]\d{3})*(?:[.,]\d+)?"
_CLAIM_RE = re.compile(rf"({_NUMBER})\s*({_UNITS})\b", re.IGNORECASE)

# Nombres écrits en toutes lettres dans le manuel, ramenés en chiffres pour
# éviter de signaler « 3 secondes » comme absent quand la source dit « trois ».
_WORDS = {
    "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "six": 6,
    "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11, "douze": 12,
    "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16, "vingt": 20,
    "trente": 30, "quarante": 40, "cinquante": 50, "soixante": 60,
    "cent": 100, "cents": 100, "mille": 1000,
}


def _strip_accents(text: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def normalize_number(raw: str) -> float | None:
    """« 1 500 » → 1500.0, « 0,08 » → 0.08. L'espace sépare les milliers,
    la virgule les décimales (convention française)."""
    cleaned = raw.replace(" ", "").replace(" ", "").replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _source_numbers(context: str) -> set[float]:
    """Toutes les valeurs numériques du manuel, chiffres et mots confondus."""
    values: set[float] = set()

    for match in re.finditer(_NUMBER, context):
        value = normalize_number(match.group(0))
        if value is not None:
            values.add(value)

    lowered = _strip_accents(context.lower())
    for word, value in _WORDS.items():
        if re.search(rf"\b{_strip_accents(word)}\b", lowered):
            values.add(float(value))

    return values


def _is_list_marker(text: str, start: int) -> bool:
    """« 1. » ou « 2) » en début de ligne numérote une liste, n'affirme rien."""
    line_start = text.rfind("\n", 0, start) + 1
    return re.fullmatch(r"[\s\-*>#]*", text[line_start:start]) is not None and (
        re.match(r"\d+\s*[.)]", text[start:start + 6]) is not None
    )


def verify_numbers(generated: str, context: str) -> list[dict]:
    """Chaque valeur chiffrée du texte généré est-elle présente dans le manuel ?"""
    source_values = _source_numbers(context)
    findings: list[dict] = []
    seen: set[tuple[float, str]] = set()

    for match in _CLAIM_RE.finditer(generated):
        if _is_list_marker(generated, match.start()):
            continue

        value = normalize_number(match.group(1))
        if value is None:
            continue

        unit = match.group(2).lower()
        if (value, unit) in seen:
            continue
        seen.add((value, unit))

        line_start = generated.rfind("\n", 0, match.start()) + 1
        line_end = generated.find("\n", match.end())
        excerpt = generated[line_start:line_end if line_end != -1 else None].strip()

        findings.append({
            "valeur": match.group(0).strip(),
            "presente_dans_le_manuel": value in source_values,
            "extrait": excerpt[:200],
        })

    return findings


# ── Étage 2 : affirmations normatives ─────────────────────────────────────────

def extract_claims(generated: str, api_key: str, limit: int = 10) -> list[str]:
    """Isole les affirmations vérifiables (obligations, interdictions, seuils)."""
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu isoles les affirmations factuelles vérifiables d'un texte "
                    "pédagogique. Réponds UNIQUEMENT en JSON valide."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Voici une fiche de révision sur le code de la route :\n\n"
                    f"{generated[:6000]}\n\n"
                    f"Extrais au plus {limit} affirmations portant sur une règle : "
                    "obligation, interdiction, seuil, priorité, sanction. "
                    "Reformule chacune en une phrase autonome et vérifiable.\n\n"
                    "Ignore les encouragements, les moyens mnémotechniques, les mises "
                    "en situation et tout ce qui relève du style.\n\n"
                    'JSON : {"affirmations": ["...", "..."]}'
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    try:
        claims = json.loads(response.choices[0].message.content).get("affirmations", [])
    except (json.JSONDecodeError, TypeError):
        return []
    return [c for c in claims if isinstance(c, str) and c.strip()][:limit]


def verify_claims(claims: list[str], api_key: str) -> list[dict]:
    """Recoupe chaque affirmation avec une recherche fraîche dans le manuel."""
    if not claims:
        return []

    # Recherche ciblée par affirmation : une règle correcte ne doit pas être
    # déclarée absente parce qu'elle manquait à la récupération d'origine.
    dossier = []
    for i, claim in enumerate(claims):
        extracts = search(claim, api_key, n_results=3)
        dossier.append(
            f"[{i}] AFFIRMATION : {claim}\n"
            f"    EXTRAITS DU MANUEL :\n    " + "\n    ".join(e[:700] for e in extracts)
        )

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es vérificateur des faits pour un organisme de formation à la "
                    "conduite. Tu juges si une affirmation est soutenue par le manuel "
                    "officiel, en te fondant EXCLUSIVEMENT sur les extraits fournis. "
                    "Tes propres connaissances ne comptent pas : si les extraits ne "
                    "disent rien, le verdict est \"absent\". Réponds UNIQUEMENT en JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{chr(10).join(dossier)}\n\n"
                    "Pour chaque affirmation, rends un verdict :\n"
                    "- \"soutenu\" : les extraits l'établissent clairement.\n"
                    "- \"contredit\" : les extraits disent autre chose. C'est grave, "
                    "un candidat apprendrait une erreur.\n"
                    "- \"absent\" : les extraits ne permettent pas de trancher.\n\n"
                    "Cite le passage exact qui fonde ton verdict (vide si absent).\n\n"
                    'JSON : {"verdicts": [{"index": 0, "verdict": "soutenu|contredit|'
                    'absent", "justification": "...", "passage_source": "..."}]}'
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        verdicts = json.loads(response.choices[0].message.content).get("verdicts", [])
    except (json.JSONDecodeError, TypeError):
        return []

    results = []
    for v in verdicts:
        idx = v.get("index")
        if isinstance(idx, int) and 0 <= idx < len(claims):
            results.append({
                "affirmation": claims[idx],
                "verdict": v.get("verdict", "absent"),
                "justification": v.get("justification", ""),
                "passage_source": v.get("passage_source", ""),
            })
    return results


# ── Rapport ───────────────────────────────────────────────────────────────────

def build_report(numbers: list[dict], claims: list[dict]) -> dict:
    chiffres_absents = [n for n in numbers if not n["presente_dans_le_manuel"]]
    contredits = [c for c in claims if c["verdict"] == "contredit"]
    absents = [c for c in claims if c["verdict"] == "absent"]

    if contredits:
        statut = "erreur"
    elif chiffres_absents or absents:
        statut = "avertissement"
    else:
        statut = "conforme"

    return {
        "statut": statut,
        "chiffres": numbers,
        "chiffres_introuvables": len(chiffres_absents),
        "chiffres_total": len(numbers),
        "affirmations": claims,
        "contredits": len(contredits),
        "absents": len(absents),
        "soutenus": len(claims) - len(contredits) - len(absents),
    }


def verify_summary(
    generated: str, theme: str, api_key: str, check_claims: bool = True
) -> dict:
    """Vérifie une fiche : chiffres toujours, affirmations si demandé."""
    context = "\n\n".join(search(theme, api_key, n_results=10))
    numbers = verify_numbers(generated, context)

    claims: list[dict] = []
    if check_claims:
        claims = verify_claims(extract_claims(generated, api_key), api_key)

    return build_report(numbers, claims)


def verify_question(question: dict, api_key: str) -> dict:
    """Vérifie qu'une question d'examen a bien la bonne réponse selon le manuel.

    C'est le contrôle le plus important de l'application : une mauvaise réponse
    marquée correcte fait apprendre l'erreur au candidat.
    """
    choices = question.get("choices", [])
    correct_index = question.get("correct_index", 0)
    if not choices or not (0 <= correct_index < len(choices)):
        return {"verdict": "invalide", "justification": "Question mal formée."}

    enonce = question.get("question", "")
    extracts = search(f"{enonce} {choices[correct_index]}", api_key, n_results=4)

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu contrôles la qualité de questions d'examen de conduite. Tu te "
                    "fondes EXCLUSIVEMENT sur les extraits du manuel officiel fournis. "
                    "Réponds UNIQUEMENT en JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"QUESTION : {enonce}\n"
                    f"CHOIX :\n" + "\n".join(f"  {i}. {c}" for i, c in enumerate(choices))
                    + f"\nRÉPONSE ANNONCÉE CORRECTE : {correct_index}\n\n"
                    f"EXTRAITS DU MANUEL :\n" + "\n\n".join(e[:800] for e in extracts)
                    + "\n\nLes extraits confirment-ils que la réponse annoncée est la "
                    "bonne ?\n"
                    "- \"valide\" : les extraits la confirment.\n"
                    "- \"mauvaise_reponse\" : les extraits désignent un autre choix. "
                    "Précise lequel.\n"
                    "- \"non_verifiable\" : les extraits ne permettent pas de trancher.\n\n"
                    'JSON : {"verdict": "valide|mauvaise_reponse|non_verifiable", '
                    '"index_correct_selon_manuel": null, "justification": "..."}'
                ),
            },
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    try:
        return json.loads(response.choices[0].message.content)
    except (json.JSONDecodeError, TypeError):
        return {"verdict": "non_verifiable", "justification": "Réponse illisible."}
