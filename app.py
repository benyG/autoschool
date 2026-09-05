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

        def _index_of(options, key, env_key, default):
            current = st.session_state.get(key) or os.getenv(env_key, "") or default
            return options.index(current) if current in options else None

        from src.models import DEFAULT_CHAT_MODEL, DEFAULT_EMBEDDING_MODEL

        if chat_options:
            choice = st.selectbox(
                "Modèle de chat",
                chat_options,
                index=_index_of(chat_options, "chat_model", "OPENAI_MODEL", DEFAULT_CHAT_MODEL),
                placeholder="Choisissez un modèle…",
            )
            if choice:
                st.session_state["chat_model"] = choice

        if embed_options:
            previous = st.session_state.get("embed_model")
            choice = st.selectbox(
                "Modèle d'embedding",
                embed_options,
                index=_index_of(embed_options, "embed_model", "OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
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
        ["🏠 Accueil", "📚 Fiches thématiques", "🎧 Bibliothèque audio",
         "🃏 Flashcards", "📝 Simulateur d'examen"],
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
    if not api_key:
        st.stop()

    from src.summaries import THEMES, generate_summary
    from src.tts import text_to_speech, strip_markdown
    from src.library import (
        save_summary, get_latest_summary, list_audios,
        save_verification, get_verification,
    )

    from src.library import verification_statuses
    statuts = verification_statuses()
    _icones = {"conforme": "✅", "avertissement": "⚠️", "erreur": "⛔"}

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        theme = st.selectbox(
            "Choisissez un thème",
            THEMES,
            format_func=lambda t: f"{_icones.get(statuts.get(t), '·')} {t}",
        )
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

    # La fiche archivée sert de point de départ : rien n'est perdu entre deux sessions.
    stored = get_latest_summary(theme)
    if stored:
        st.caption(f"Fiche enregistrée le {stored['created_at'].replace('T', ' à ')}")

    col_gen, col_tts, col_check = st.columns(3)
    label = "🔄 Régénérer la fiche" if stored else "📖 Générer la fiche"
    generate = col_gen.button(label, use_container_width=True, type="primary")
    read_aloud = col_tts.button("🔊 Écouter", use_container_width=True, disabled=not stored)
    check = col_check.button(
        "🔍 Vérifier", use_container_width=True, disabled=not stored,
        help="Recoupe les chiffres et les règles de la fiche avec le manuel officiel.",
    )

    if generate:
        from src.vector_store import is_indexed
        if not is_indexed(api_key):
            st.error("Indexez d'abord les PDFs (barre latérale).")
            st.stop()
        try:
            with st.spinner(f"Analyse des enjeux d'examen puis rédaction — « {theme} »…"):
                summary, points = generate_summary(theme, api_key)
            save_summary(theme, summary, st.session_state.get("chat_model", ""))
            st.session_state[f"points_{theme}"] = points
            st.rerun()
        except Exception as e:
            st.error(f"Génération impossible : {e}")

    if stored:
        summary_text = stored["content"]

        points = st.session_state.get(f"points_{theme}", [])
        if points:
            counts = {n: sum(1 for p in points if p.get("criticite") == n)
                      for n in ("critique", "important", "secondaire")}
            c1, c2, c3 = st.columns(3)
            c1.metric("🔴 Points critiques", counts["critique"])
            c2.metric("🟠 Points importants", counts["important"])
            c3.metric("⚪ Bon à savoir", counts["secondaire"])

        if check:
            try:
                with st.spinner("Recoupement avec le manuel officiel…"):
                    from src.verification import verify_summary
                    report = verify_summary(summary_text, theme, api_key)
                save_verification(stored["hash"], theme, report)
                st.rerun()
            except Exception as e:
                st.error(f"Vérification impossible : {e}")

        # Le rapport porte sur ce contenu précis : régénérer la fiche l'invalide.
        report = get_verification(stored["hash"])
        if report:
            statut = report["statut"]
            if statut == "erreur":
                st.error(
                    f"⛔ **{report['contredits']} affirmation(s) contredites par le "
                    "manuel.** Ne révise pas cette fiche en l'état — régénère-la."
                )
            elif statut == "avertissement":
                details = []
                if report["chiffres_introuvables"]:
                    details.append(f"{report['chiffres_introuvables']} chiffre(s) introuvables")
                if report["absents"]:
                    details.append(f"{report['absents']} règle(s) non retrouvées")
                st.warning(f"⚠️ À confirmer : {', '.join(details)}.")
            else:
                st.success(
                    f"✅ Fiche recoupée avec le manuel — "
                    f"{report['chiffres_total']} chiffre(s) et {report['soutenus']} "
                    "règle(s) confirmés."
                )

            with st.expander("Détail de la vérification"):
                st.caption(f"Vérifiée le {report['verifie_le'].replace('T', ' à ')}")

                if report["chiffres"]:
                    st.markdown("**Chiffres**")
                    for c in report["chiffres"]:
                        icon = "✅" if c["presente_dans_le_manuel"] else "❓"
                        st.markdown(f"{icon} `{c['valeur']}` — {c['extrait']}")
                        if not c["presente_dans_le_manuel"]:
                            st.caption("Cette valeur n'apparaît pas dans les extraits du manuel.")

                if report["affirmations"]:
                    st.markdown("**Règles**")
                    icons = {"soutenu": "✅", "contredit": "⛔", "absent": "❓"}
                    for a in report["affirmations"]:
                        st.markdown(f"{icons.get(a['verdict'], '❓')} {a['affirmation']}")
                        if a.get("justification"):
                            st.caption(a["justification"])
                        if a.get("passage_source"):
                            st.caption(f"› Manuel : « {a['passage_source'][:300]} »")
        elif stored:
            st.caption("Fiche non vérifiée — clique sur « Vérifier » pour la recouper avec le manuel.")

        with st.container(border=True):
            st.markdown(summary_text)

        if read_aloud:
            try:
                bar = st.progress(0.0, text="Préparation de l'audio…")

                def _avance(fait: int, total: int):
                    bar.progress(fait / total, text=f"Synthèse — partie {fait}/{total}")

                audio = text_to_speech(
                    summary_text, api_key, voice=voice, theme=theme, on_progress=_avance
                )
                bar.empty()
                st.session_state[f"audio_{theme}"] = audio
            except Exception as e:
                st.error(f"Synthèse vocale indisponible : {e}")

        # Audios déjà archivés pour ce thème (toutes voix confondues).
        archived = list_audios(theme)
        current = st.session_state.get(f"audio_{theme}")
        to_play = current or (archived[0] if archived else None)

        if to_play:
            from src.audio import format_duration, mime_for
            from pathlib import Path as _Path

            st.markdown("#### 🎧 Écoute")
            with open(to_play["path"], "rb") as f:
                data = f.read()
            mime = mime_for(to_play["path"])
            st.audio(data, format=mime)

            # La piste doit couvrir tout le texte : on le vérifie au lieu de
            # laisser l'élève s'en apercevoir en cours d'écoute.
            duree = to_play.get("duration_seconds")
            attendu = len(strip_markdown(summary_text)) / 15.0  # ~15 caractères/s
            if duree and attendu and duree < attendu * 0.75:
                st.warning(
                    f"⚠️ Cette piste dure {format_duration(duree)} alors que le texte "
                    f"en demande environ {format_duration(attendu)}. Elle date "
                    "probablement d'une version antérieure — régénère-la."
                )

            meta = st.columns([3, 1])
            meta[0].caption(
                f"Voix **{to_play['voice']}** · {to_play['model']} · "
                f"**{format_duration(duree)}** · "
                f"{to_play['size_bytes'] / 1_000_000:.1f} Mo · "
                f"généré le {to_play['created_at'].replace('T', ' à ')}"
            )
            meta[1].download_button(
                "⬇️ Télécharger",
                data,
                file_name=f"{theme.replace(' ', '_')}_{to_play['voice']}"
                          f"{_Path(to_play['path']).suffix}",
                mime=mime,
                use_container_width=True,
            )
            if len(archived) > 1:
                st.caption(
                    f"{len(archived)} versions archivées pour ce thème — "
                    "retrouve-les toutes dans la Bibliothèque audio."
                )


# BIBLIOTHÈQUE AUDIO
elif page == "🎧 Bibliothèque audio":
    st.title("🎧 Bibliothèque audio")
    from src.library import list_audios, delete_audio, library_stats
    from src.audio import format_duration, mime_for, ffmpeg_available

    stats = library_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Audios archivés", stats["audio_count"])
    c2.metric("Écoute totale", format_duration(stats["audio_seconds"]))
    c3.metric("Espace occupé", f"{stats['audio_bytes'] / 1_000_000:.1f} Mo")
    c4.metric("Fiches enregistrées", stats["summary_count"])

    if not ffmpeg_available():
        st.caption(
            "💡 Les pistes sont enregistrées en WAV (environ 3 Mo par minute). "
            "Installer **ffmpeg** et le rendre accessible dans le PATH les fera "
            "compresser automatiquement en MP3, dix fois plus légères."
        )

    audios = list_audios()
    if not audios:
        st.info(
            "Aucun audio pour l'instant. Génère une fiche puis clique sur « Écouter » : "
            "le fichier sera conservé ici et réutilisé sans être régénéré."
        )
    else:
        st.caption(
            "Les fichiers sont dans `data/audio/`. Un même texte lu avec la même voix "
            "réutilise l'audio existant plutôt que de le repayer."
        )
        themes = sorted({a["theme"] for a in audios})
        chosen = st.multiselect("Filtrer par thème", themes, default=[])
        shown = [a for a in audios if not chosen or a["theme"] in chosen]

        from pathlib import Path as _Path

        for audio in shown:
            with st.expander(
                f"{audio['theme']} — voix {audio['voice']} · "
                f"{format_duration(audio.get('duration_seconds'))} "
                f"({audio['created_at'].replace('T', ' à ')})"
            ):
                with open(audio["path"], "rb") as f:
                    data = f.read()
                mime = mime_for(audio["path"])
                st.audio(data, format=mime)
                cols = st.columns([2, 1, 1])
                cols[0].caption(
                    f"{audio['model']} · {audio['size_bytes'] / 1_000_000:.1f} Mo · "
                    f"{_Path(audio['path']).suffix.lstrip('.').upper()}"
                )
                cols[1].download_button(
                    "⬇️ Télécharger",
                    data,
                    file_name=f"{audio['theme'].replace(' ', '_')}_{audio['voice']}"
                              f"{_Path(audio['path']).suffix}",
                    mime=mime,
                    use_container_width=True,
                    key=f"dl_{audio['hash']}",
                )
                if cols[2].button(
                    "🗑️ Supprimer", use_container_width=True, key=f"del_{audio['hash']}"
                ):
                    delete_audio(audio["hash"])
                    st.rerun()


# FLASHCARDS
elif page == "🃏 Flashcards":
    st.title("🃏 Flashcards — Répétition espacée")
    api_key = require_api_key()
    if not api_key:
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
    if not api_key:
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
        verify = st.checkbox(
            "🔍 Vérifier chaque question contre le manuel avant de commencer",
            value=True,
            help="Écarte les questions dont la bonne réponse n'est pas confirmée par "
                 "le manuel. Plus lent, mais évite d'apprendre une erreur.",
        )

        if st.button("🚀 Démarrer l'examen", use_container_width=True, type="primary"):
            with st.spinner("Génération de l'examen… (30-60 secondes)"):
                questions = generate_exam(api_key, n_questions=20)

            ecartees = []
            if verify and questions:
                from src.verification import verify_question
                progress = st.progress(0.0, text="Vérification des questions…")
                retenues = []
                for i, q in enumerate(questions):
                    try:
                        verdict = verify_question(q, api_key)
                    except Exception:
                        # Un échec de vérification ne doit pas priver d'examen.
                        retenues.append(q)
                        progress.progress((i + 1) / len(questions))
                        continue

                    if verdict.get("verdict") == "mauvaise_reponse":
                        # Le manuel désigne un autre choix : on corrige plutôt
                        # que de jeter, si l'index proposé est exploitable.
                        idx = verdict.get("index_correct_selon_manuel")
                        if isinstance(idx, int) and 0 <= idx < len(q.get("choices", [])):
                            q["correct_index"] = idx
                            q["explanation"] = (
                                verdict.get("justification") or q.get("explanation", "")
                            )
                            retenues.append(q)
                        else:
                            ecartees.append(q)
                    else:
                        retenues.append(q)
                    progress.progress((i + 1) / len(questions))
                progress.empty()
                questions = retenues

            if not questions:
                st.error(
                    "Aucune question n'a pu être validée contre le manuel. "
                    "Vérifie que l'index couvre bien ce contenu, puis réessaie."
                )
                st.stop()

            st.session_state["exam_ecartees"] = len(ecartees)
            st.session_state["exam_questions"] = questions
            st.session_state["exam_answers"] = [-1] * len(questions)
            st.session_state["exam_state"] = "in_progress"
            st.rerun()

    elif state == "in_progress":
        questions = st.session_state["exam_questions"]
        answers = st.session_state["exam_answers"]

        ecartees = st.session_state.get("exam_ecartees", 0)
        if ecartees:
            st.info(
                f"{ecartees} question(s) écartée(s) : le manuel ne confirmait pas "
                f"leur réponse. L'examen en compte {len(questions)}."
            )

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
