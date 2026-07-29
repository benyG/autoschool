import streamlit as st
import os
import time
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AutoÉcole QC",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.card {
    background: var(--background-color, rgba(255,255,255,0.05));
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.score-pass { color: #28a745; font-size: 2rem; font-weight: bold; }
.score-fail { color: #dc3545; font-size: 2rem; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ── API Key ───────────────────────────────────────────────────────────────────
def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        key = st.session_state.get("openai_key", "")
    return key or None


def require_api_key() -> str | None:
    key = get_api_key()
    if not key:
        st.warning("Entrez votre clé OpenAI dans la barre latérale pour continuer.")
        return None
    return key


def require_chat_model() -> bool:
    from src.models import get_chat_model, ModelNotConfigured
    try:
        get_chat_model()
        return True
    except ModelNotConfigured as e:
        st.warning(str(e))
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 AutoÉcole QC")
    st.markdown("---")

    if not os.getenv("OPENAI_API_KEY"):
        st.subheader("🔑 Clé OpenAI")
        key_input = st.text_input("API Key", type="password", key="key_input")
        if key_input:
            st.session_state["openai_key"] = key_input
            st.success("Clé enregistrée !")

    st.markdown("---")
    st.subheader("🤖 Modèles OpenAI")

    api_key = get_api_key()
    if not api_key:
        st.caption("Entrez votre clé pour voir les modèles disponibles.")
    else:
        if "available_models" not in st.session_state:
            with st.spinner("Lecture des modèles disponibles…"):
                try:
                    from src.models import list_available_models
                    st.session_state["available_models"] = list_available_models(api_key)
                except Exception as e:
                    st.session_state["available_models"] = ([], [])
                    st.error(f"Impossible de lister les modèles : {e}")

        chat_options, embed_options = st.session_state["available_models"]

        def _index_of(options, key, env_key):
            current = st.session_state.get(key) or os.getenv(env_key, "")
            return options.index(current) if current in options else None

        if chat_options:
            choice = st.selectbox(
                "Modèle de chat",
                chat_options,
                index=_index_of(chat_options, "chat_model", "OPENAI_MODEL"),
                placeholder="Choisissez un modèle…",
            )
            if choice:
                st.session_state["chat_model"] = choice

        if embed_options:
            previous = st.session_state.get("embed_model")
            choice = st.selectbox(
                "Modèle d'embedding",
                embed_options,
                index=_index_of(embed_options, "embed_model", "OPENAI_EMBEDDING_MODEL"),
                placeholder="Choisissez un modèle…",
            )
            if choice:
                st.session_state["embed_model"] = choice
                if previous and previous != choice:
                    st.warning(
                        "Le modèle d'embedding a changé : réindexez les documents "
                        "(bouton « Réinitialiser l'index » ci-dessous)."
                    )

        if st.button("🔄 Rafraîchir la liste", use_container_width=True):
            st.session_state.pop("available_models", None)
            st.rerun()

    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Accueil", "📚 Fiches thématiques", "🃏 Flashcards", "📝 Simulateur d'examen"],
        label_visibility="collapsed",
    )

    # Index status
    st.markdown("---")
    st.subheader("⚙️ Base de documents")
    if api_key:
        from src.vector_store import is_indexed, reset_index
        if is_indexed(api_key):
            st.success("Documents indexés ✓")
            if st.button("♻️ Réinitialiser l'index", use_container_width=True):
                reset_index()
                st.rerun()
        else:
            st.info("Documents non indexés")
            if st.button("Indexer les PDFs", use_container_width=True, type="primary"):
                try:
                    with st.spinner("Extraction et indexation en cours…"):
                        from src.pdf_parser import load_all_docs
                        from src.vector_store import index_chunks
                        _, chunks = load_all_docs("doc")
                        index_chunks(chunks, api_key)
                    st.success(f"Indexé ({len(chunks)} passages)")
                    st.rerun()
                except Exception as e:
                    st.error(f"Échec de l'indexation : {e}")


# ── Pages ─────────────────────────────────────────────────────────────────────

# HOME
if page == "🏠 Accueil":
    st.title("🚗 AutoÉcole Québec — Votre préparateur IA")
    st.markdown("""
    Bienvenue ! Cette application vous aide à préparer votre examen de code de la route
    de la **SAAQ** à partir du manuel officiel du Québec.

    ### Comment commencer ?
    1. **Entrez votre clé OpenAI** dans la barre latérale
    2. **Indexez les PDFs** (bouton dans la barre latérale) — une seule fois
    3. Choisissez votre mode d'apprentissage :

    | Mode | Description |
    |------|-------------|
    | 📚 Fiches thématiques | Résumés clairs + lecture à voix haute |
    | 🃏 Flashcards | Répétition espacée (algorithme SM-2) |
    | 📝 Simulateur d'examen | 20 questions, conditions réelles SAAQ |
    """)

    api_key = get_api_key()
    if api_key:
        from src.flashcards import get_stats
        stats = get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Flashcards créées", stats["total"])
        col2.metric("À réviser aujourd'hui", stats["due"])
        col3.metric("Maîtrisées", stats["mastered"])


# FICHES THÉMATIQUES
elif page == "📚 Fiches thématiques":
    st.title("📚 Fiches thématiques")
    api_key = require_api_key()
    if not api_key or not require_chat_model():
        st.stop()

    from src.summaries import THEMES, generate_summary
    from src.tts import text_to_speech, cleanup_audio

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        theme = st.selectbox("Choisissez un thème", THEMES)
    with col2:
        voice = st.selectbox("Voix TTS", ["nova", "alloy", "echo", "fable", "onyx", "shimmer"])
    with col3:
        if "tts_models" not in st.session_state:
            from src.models import list_tts_models
            try:
                st.session_state["tts_models"] = list_tts_models(api_key)
            except Exception:
                st.session_state["tts_models"] = []
        tts_options = st.session_state["tts_models"]
        tts_choice = st.selectbox(
            "Modèle TTS",
            tts_options or ["(aucun disponible)"],
            disabled=not tts_options,
        )
        if tts_options:
            st.session_state["tts_model"] = tts_choice

    col_gen, col_tts = st.columns(2)
    generate = col_gen.button("📖 Générer la fiche", use_container_width=True, type="primary")
    read_aloud = col_tts.button("🔊 Lire à voix haute", use_container_width=True)

    cache_key = f"summary_{theme}"

    if generate:
        from src.vector_store import is_indexed
        if not is_indexed(api_key):
            st.error("Indexez d'abord les PDFs (barre latérale).")
            st.stop()
        with st.spinner(f"Génération de la fiche '{theme}'…"):
            summary = generate_summary(theme, api_key)
            st.session_state[cache_key] = summary

    if cache_key in st.session_state:
        summary_text = st.session_state[cache_key]
        st.markdown(f"""<div class="card">{summary_text}</div>""", unsafe_allow_html=True)

        if read_aloud:
            try:
                with st.spinner("Synthèse vocale en cours…"):
                    audio_path = text_to_speech(summary_text, api_key, voice=voice)
                with open(audio_path, "rb") as f:
                    st.audio(f.read(), format="audio/mp3")
                cleanup_audio(audio_path)
            except Exception as e:
                st.error(f"Synthèse vocale indisponible : {e}")


# FLASHCARDS
elif page == "🃏 Flashcards":
    st.title("🃏 Flashcards — Répétition espacée")
    api_key = require_api_key()
    if not api_key or not require_chat_model():
        st.stop()

    from src.flashcards import (
        get_due_cards, review_card, generate_flashcards,
        save_flashcards, get_stats
    )
    from src.summaries import THEMES

    tab1, tab2 = st.tabs(["Réviser", "Créer des flashcards"])

    with tab1:
        stats = get_stats()
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", stats["total"])
        col2.metric("À réviser", stats["due"])
        col3.metric("Maîtrisées", stats["mastered"])

        if stats["due"] == 0:
            st.success("🎉 Aucune flashcard à réviser pour aujourd'hui ! Revenez demain.")
        else:
            if "fc_queue" not in st.session_state or st.button("Commencer la session"):
                st.session_state["fc_queue"] = get_due_cards(20)
                st.session_state["fc_index"] = 0
                st.session_state["fc_revealed"] = False

            queue = st.session_state.get("fc_queue", [])
            idx = st.session_state.get("fc_index", 0)

            if idx < len(queue):
                card = queue[idx]
                progress = idx / len(queue)
                st.progress(progress, text=f"Carte {idx + 1} / {len(queue)}")

                st.markdown(f"""
                <div class="card">
                    <p style="color:#666; font-size:0.85rem;">Thème : {card.get('theme','')}</p>
                    <h3>❓ {card['question']}</h3>
                </div>
                """, unsafe_allow_html=True)

                if not st.session_state.get("fc_revealed"):
                    if st.button("Révéler la réponse", use_container_width=True, type="primary"):
                        st.session_state["fc_revealed"] = True
                        st.rerun()
                else:
                    st.markdown(f"""
                    <div class="card" style="border-left: 4px solid #28a745;">
                        <h4>✅ Réponse</h4>
                        <p>{card['answer']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("**Comment ça s'est passé ?**")
                    cols = st.columns(4)
                    labels = [("❌ Raté", 0), ("😓 Difficile", 1), ("🙂 Ok", 2), ("😎 Facile", 3)]
                    for col, (label, quality) in zip(cols, labels):
                        if col.button(label, use_container_width=True):
                            review_card(card["id"], quality)
                            st.session_state["fc_index"] += 1
                            st.session_state["fc_revealed"] = False
                            st.rerun()
            else:
                st.success("🎉 Session terminée ! Toutes les cartes ont été révisées.")
                if st.button("Recommencer"):
                    del st.session_state["fc_queue"]
                    st.rerun()

    with tab2:
        from src.vector_store import is_indexed
        if not is_indexed(api_key):
            st.error("Indexez d'abord les PDFs (barre latérale).")
            st.stop()

        theme = st.selectbox("Thème", THEMES, key="fc_theme")
        count = st.slider("Nombre de flashcards à générer", 4, 15, 8)

        if st.button("Générer et sauvegarder", type="primary"):
            with st.spinner(f"Génération de {count} flashcards sur '{theme}'…"):
                cards = generate_flashcards(theme, api_key, count)
                save_flashcards(cards, theme)
            st.success(f"{len(cards)} flashcards ajoutées à votre deck !")
            for c in cards:
                with st.expander(c["question"]):
                    st.write(c["answer"])


# SIMULATEUR D'EXAMEN
elif page == "📝 Simulateur d'examen":
    st.title("📝 Simulateur d'examen SAAQ")
    api_key = require_api_key()
    if not api_key or not require_chat_model():
        st.stop()

    from src.quiz import generate_exam, evaluate_exam
    from src.vector_store import is_indexed

    if not is_indexed(api_key):
        st.error("Indexez d'abord les PDFs (barre latérale).")
        st.stop()

    # State machine: idle → in_progress → results
    state = st.session_state.get("exam_state", "idle")

    if state == "idle":
        st.markdown("""
        ### Conditions de l'examen
        - **20 questions** à choix multiples
        - Tirage aléatoire parmi tous les thèmes
        - **Seuil de réussite : 80%** (comme la SAAQ)
        - Pas de limite de temps ici, mais travaillez vite !
        """)
        if st.button("🚀 Démarrer l'examen", use_container_width=True, type="primary"):
            with st.spinner("Génération de l'examen… (30-60 secondes)"):
                questions = generate_exam(api_key, n_questions=20)
            st.session_state["exam_questions"] = questions
            st.session_state["exam_answers"] = [-1] * len(questions)
            st.session_state["exam_state"] = "in_progress"
            st.rerun()

    elif state == "in_progress":
        questions = st.session_state["exam_questions"]
        answers = st.session_state["exam_answers"]

        answered = sum(1 for a in answers if a >= 0)
        st.progress(answered / len(questions), text=f"{answered}/{len(questions)} répondues")

        for i, q in enumerate(questions):
            with st.container():
                st.markdown(f"**Q{i+1}.** {q['question']}")
                choices = q.get("choices", [])
                current = answers[i] if answers[i] >= 0 else None
                selected = st.radio(
                    f"q{i}",
                    options=range(len(choices)),
                    format_func=lambda x, c=choices: c[x],
                    index=current,
                    label_visibility="collapsed",
                    key=f"exam_q{i}",
                )
                if selected is not None:
                    st.session_state["exam_answers"][i] = selected
                st.markdown("---")

        all_answered = all(a >= 0 for a in st.session_state["exam_answers"])
        col1, col2 = st.columns(2)
        if col1.button("✅ Terminer l'examen", disabled=not all_answered, type="primary", use_container_width=True):
            results = evaluate_exam(questions, st.session_state["exam_answers"])
            st.session_state["exam_results"] = results
            st.session_state["exam_state"] = "results"
            st.rerun()
        if col2.button("❌ Abandonner", use_container_width=True):
            st.session_state["exam_state"] = "idle"
            st.rerun()

        if not all_answered:
            remaining = sum(1 for a in st.session_state["exam_answers"] if a < 0)
            st.warning(f"Répondez à toutes les questions ({remaining} restantes).")

    elif state == "results":
        results = st.session_state["exam_results"]
        pct = results["percentage"]
        passed = results["passed"]

        score_class = "score-pass" if passed else "score-fail"
        badge = "RÉUSSI ✅" if passed else "ÉCHOUÉ ❌"
        st.markdown(f"""
        <div class="card" style="text-align:center;">
            <p class="{score_class}">{pct}% — {badge}</p>
            <p>{results['score']} / {results['total']} bonnes réponses</p>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Détail des réponses")
        for i, r in enumerate(results["results"]):
            icon = "✅" if r["is_correct"] else "❌"
            color = "#d4edda" if r["is_correct"] else "#f8d7da"
            with st.expander(f"{icon} Q{i+1} — {r['question'][:70]}…"):
                for j, choice in enumerate(r["choices"]):
                    prefix = ""
                    if j == r["correct_answer"]:
                        prefix = "✅ "
                    elif j == r["user_answer"] and not r["is_correct"]:
                        prefix = "❌ "
                    st.write(f"{prefix}{choice}")
                if r.get("explanation"):
                    st.info(f"💡 {r['explanation']}")
                st.caption(f"Thème : {r['theme']}")

        if st.button("🔄 Nouvel examen", use_container_width=True, type="primary"):
            st.session_state["exam_state"] = "idle"
            st.rerun()
