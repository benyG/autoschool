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


def generate_summary(theme: str, openai_api_key: str) -> str:
    client = OpenAI(api_key=openai_api_key)
    context_chunks = search(theme, openai_api_key, n_results=6)
    context = "\n\n".join(context_chunks)

    response = client.chat.completions.create(
        model=get_chat_model(),
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un moniteur d'auto-école québécois chevronné, du genre que tout le "
                    "monde recommande parce qu'il rend les choses limpides. Tu expliques le code "
                    "de la route à un élève assis à côté de toi, pas à une salle de classe.\n\n"
                    "Ta façon de parler :\n"
                    "- Tutoie l'élève, parle-lui directement (« tu », « regarde », « imagine que »).\n"
                    "- Écris comme tu parles : phrases courtes, ton chaleureux, zéro jargon "
                    "administratif. Si un terme technique est indispensable, explique-le "
                    "aussitôt avec tes mots.\n"
                    "- Explique toujours le POURQUOI derrière une règle avant le QUOI. Une règle "
                    "qu'on comprend, on ne l'oublie plus.\n"
                    "- Ancre chaque notion dans une situation concrète de conduite au Québec "
                    "(un stop enneigé à Québec, un boulevard à Montréal, une sortie d'autoroute).\n"
                    "- Bannis les formules robotiques : pas de « Il est important de noter que », "
                    "pas de « En conclusion », pas d'énumération sèche de règlements.\n"
                    "- Anticipe la confusion : « Là, beaucoup de monde se trompe… », "
                    "« Attention, ça ressemble à X mais c'est différent parce que… ».\n\n"
                    "Le format doit rester une fiche de révision : titres clairs, paragraphes "
                    "courts, listes quand ça aide vraiment, chiffres clés en gras. Mais le texte "
                    "entre les puces doit sonner comme une vraie explication humaine, pas comme "
                    "un extrait de règlement recopié."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Explique-moi le thème : **{theme}**\n\n"
                    f"Appuie-toi sur ces extraits du manuel officiel du Québec :\n\n{context}\n\n"
                    "Déroule ton explication comme une vraie leçon :\n"
                    "1. Ouvre avec une phrase qui me dit pourquoi ce sujet compte pour moi "
                    "concrètement au volant.\n"
                    "2. Explique les règles essentielles en me faisant comprendre leur logique, "
                    "pas en me les récitant.\n"
                    "3. Donne-moi les chiffres à connaître par cœur (vitesses, distances, délais) "
                    "en gras, avec un repère concret pour chacun quand c'est possible.\n"
                    "4. Préviens-moi des pièges où les gens se plantent le plus, en expliquant "
                    "pourquoi on s'y fait avoir.\n"
                    "5. Termine par un truc mnémotechnique ou une image mentale qui colle.\n\n"
                    "Reste fidèle au contenu du manuel : n'invente aucune règle ni aucun chiffre. "
                    "Si les extraits ne couvrent pas un point, dis-le franchement plutôt que "
                    "de combler le vide."
                ),
            },
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content
