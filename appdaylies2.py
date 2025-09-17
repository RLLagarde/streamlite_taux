# import streamlit as st

# # ---- Gate d’accès (facultatif mais recommandé) ----
# gate = st.secrets.get("APP_PASSWORD", "")
# if gate:
#     pw = st.text_input("Mot de passe", type="password")
#     if pw != gate:
#         st.stop()

# from __future__ import annotations
# import os, re, json, imaplib, email, io
# from email.header import decode_header, make_header
# from email.utils import parsedate_to_datetime
# from pathlib import Path
# from datetime import date, datetime, timedelta
# from typing import Dict, List, Tuple

# # (tu as déjà importé Streamlit ci-dessus)
# from dateutil.relativedelta import relativedelta

# # =========================
# # CONFIG
# # =========================
# st.set_page_config(page_title="Weekly / Monthly – Hybrids Views", layout="wide")

# # ⚠️ Ne mets PAS de secrets en dur ; on lit depuis st.secrets
# GMAIL_ADDRESS       = st.secrets.get("GMAIL_ADDRESS", "")
# GMAIL_APP_PASSWORD  = st.secrets.get("GMAIL_APP_PASSWORD", "")
# OPENAI_API_KEY      = st.secrets.get("OPENAI_API_KEY", "")
# OPENAI_MODEL        = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# SUBJECT_WEEKLY_FILTER  = st.secrets.get("SUBJECT_WEEKLY_FILTER",  "Hybrids Financial Debts Views")
# SUBJECT_MONTHLY_FILTER = st.secrets.get("SUBJECT_MONTHLY_FILTER", "Hybrids Financial Debts Views Weekly")

# GMAIL_SEARCH_DAYS = int(st.secrets.get("GMAIL_SEARCH_DAYS", 30))
# MAX_EMAILS_FETCH  = int(st.secrets.get("MAX_EMAILS_FETCH", 400))

# DATA_DIR    = Path("data")
# WEEKLY_DIR  = DATA_DIR / "weekly"
# MONTHLY_DIR = DATA_DIR / "monthly"
# for d in (DATA_DIR, WEEKLY_DIR, MONTHLY_DIR): d.mkdir(parents=True, exist_ok=True)



# # Rubriques (ordre thématique)
# DAILY_FIELDS = ["ma_and_ratings","results","financial_credit_spreads","primary_market","other_important_infos"]

# # =========================
# # HELPERS FS/FORMAT
# # =========================
# def weekly_path(w: str) -> Path:   return WEEKLY_DIR / f"{w}.json"
# def monthly_path(m: str) -> Path:  return MONTHLY_DIR / f"{m}.json"

# def load_json(p: Path, default):
#     if p.exists():
#         try: return json.loads(p.read_text(encoding="utf-8"))
#         except: return default
#     return default

# def save_json(p: Path, payload): p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# def iso_week_key(d: date) -> str:
#     y, w, _ = d.isocalendar()
#     return f"{y}-W{w:02d}"

# def monday_sunday(d: date) -> Tuple[date, date]:
#     """Retourne le lundi/dimanche de la semaine du jour donné."""
#     monday = d - timedelta(days=d.weekday())
#     sunday = monday + timedelta(days=6)
#     return monday, sunday

# def md_block(text: str) -> str:
#     return (text or "").strip().replace("\n", "  \n") or "_(vide)_"

# def normalize_line(s: str) -> str:
#     return re.sub(r"\s+", " ", s.strip().strip("•-–·* ")).lower()

# def dedupe_bullets(text: str) -> str:
#     seen, out = set(), []
#     for ln in (text or "").splitlines():
#         base = normalize_line(ln)
#         if base and base not in seen:
#             seen.add(base)
#             out.append(f"- {ln.strip().strip('•-–·* ')}")
#     return "\n".join(out)

# def summarize_texts(texts: List[str], max_bullets=12) -> str:
#     tokens = []
#     for t in texts or []:
#         for ln in str(t or "").splitlines():
#             s = ln.strip(" •-–\t ")
#             if s: tokens.append(s)
#     uniq = list(dict.fromkeys(tokens))[:max_bullets]
#     return "\n".join(f"- {b}" for b in uniq)

# # =========================
# # GMAIL
# # =========================
# def _decode_header(s) -> str:
#     try: return str(make_header(decode_header(s or "")))
#     except: return s or ""

# def _strip_html(html: str) -> str:
#     t = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
#     # IMPORTANT: pas de \n forcés sur <br> / </p> -> on laisse le LLM recoller les wraps
#     t = re.sub(r"(?s)<.*?>", " ", t)
#     t = re.sub(r"[ \t]+", " ", t)
#     t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
#     return t.strip()

# def _message_to_text(msg: email.message.Message) -> str:
#     if msg.is_multipart():
#         # 1) text/plain prioritaire
#         for part in msg.walk():
#             if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
#                 try: return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
#                 except: pass
#         # 2) text/html
#         for part in msg.walk():
#             if part.get_content_type() == "text/html" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
#                 try: return _strip_html(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
#                 except: pass
#         # 3) fallback: concat de tous les text/*
#         payloads=[]
#         for part in msg.walk():
#             if part.get_content_maintype()=="text":
#                 try: payloads.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
#                 except: pass
#         return "\n\n".join(payloads)
#     else:
#         try:
#             payload = msg.get_payload(decode=True)
#             if payload is None: return str(msg.get_payload())
#             text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
#             return _strip_html(text) if msg.get_content_type()=="text/html" else text
#         except: return str(msg.get_payload())

# def fetch_gmail_messages(newer_than_days=30, limit=400) -> List[dict]:
#     """Renvoie [{subject, body, dt, message_id}] sans filtre d'objet; on filtrera ensuite en Python."""
#     res=[]
#     if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
#         return res
#     try:
#         imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
#         imap.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
#         ok,_ = imap.select("INBOX")
#         if ok!="OK": raise RuntimeError("INBOX non sélectionnable")
#         since = (datetime.utcnow() - timedelta(days=newer_than_days)).strftime("%d-%b-%Y")
#         ok, data = imap.search(None, f'(SINCE {since})')
#         if ok!="OK": return res
#         ids = (data[0].split() or [])[-limit:]
#         for msg_id in reversed(ids):
#             ok, msg_data = imap.fetch(msg_id, "(RFC822)")
#             if ok!="OK": continue
#             msg  = email.message_from_bytes(msg_data[0][1])
#             subj = _decode_header(msg.get("Subject",""))
#             body = _message_to_text(msg)
#             try:
#                 dt = parsedate_to_datetime(msg.get("Date"))
#                 if dt and dt.tzinfo: dt = dt.astimezone().replace(tzinfo=None)
#             except: 
#                 dt = datetime.now()
#             mid = (msg.get("Message-Id") or msg.get("Message-ID") or "").strip() or f"INBOX:{msg_id.decode()}"
#             res.append({"subject": subj, "body": body, "dt": dt, "message_id": mid})
#         imap.logout()
#     except Exception as e:
#         st.error(f"Erreur Gmail IMAP: {e}")
#     return res

# # =========================
# # CLEAN BODY (quotes/signatures)
# # =========================
# QUOTE_PATTERNS = [
#     r"^On .+ wrote:$", r"^Le .+ a écrit :", r"^From: .+$", r"^-{2,}\s*Original Message\s*-{2,}$",
#     r"^> .*", r"^__+$", r"^\s*De : .+$", r"^\s*Envoyé : .+$"
# ]
# def strip_quotes_and_signatures(txt: str) -> str:
#     lines, out = (txt or "").splitlines(), []
#     for ln in lines:
#         if any(re.match(p, ln.strip(), re.IGNORECASE) for p in QUOTE_PATTERNS): break
#         out.append(ln)
#     joined = "\n".join(out)
#     joined = re.sub(r"(?is)this message.*?confidential.*", "", joined)
#     joined = re.sub(r"(?is)ce message.*?confidentiel.*", "", joined)
#     return joined.strip()

# # =========================
# # PDF LOADER
# # =========================
# def read_pdf_bytes_to_text(data: bytes) -> str:
#     """Lecture simple PDF -> texte (PyPDF2)."""
#     try:
#         import PyPDF2
#         reader = PyPDF2.PdfReader(io.BytesIO(data))
#         pages = []
#         for p in reader.pages:
#             try:
#                 pages.append(p.extract_text() or "")
#             except:
#                 pages.append("")
#         return "\n".join(pages)
#     except Exception as e:
#         st.error(f"Lecture PDF: {e}")
#         return ""

# def guess_date_from_filename(name: str) -> datetime|None:
#     # essaie YYYY-MM-DD dans le nom de fichier
#     m = re.search(r"(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])", name)
#     if not m: return None
#     try:
#         s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
#         return datetime.fromisoformat(s)
#     except:
#         return None

# # =========================
# # OPENAI (Classifier + Synthèses)
# # =========================
# from openai import OpenAI
# client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# # ---- DAILY CLASSIFIER PROMPT ---- (réutilisé pour chaque document)
# DAILY_SYS = (
#     "Tu es un assistant financier. Classe TOUT le contenu pertinent du document dans EXACTEMENT ces 5 catégories.\n"
#     "Aucune paraphrase. Recopie le texte source tel quel, en phrases complètes (jamais tronquées).\n"
#     "1 item = 1 phrase autonome et grammaticale (pas de virgule finale/parenthèse ouverte).\n"
#     "Si des retours à la ligne coupent une phrase, RECOLLE les segments pour former 1 phrase complète.\n"
#     "Exclure signatures/politesse/disclaimers.\n"
#     "En français."
# )
# DAILY_USER_TMPL = """Texte :
# ---
# [[EMAIL_BODY]]
# ---
# Retourne un JSON strict:
# {
#   "financial_credit_spreads": ["..."],
#   "primary_market": ["..."],
#   "results": ["..."],
#   "ma_and_ratings": ["..."],
#   "other_important_infos": ["..."]
# }"""

# # ---- WEEKLY SYNTH PROMPT ----
# WEEKLY_SYS = (
#     "Tu es analyste crédit. À partir d’un contenu agrégé (déjà classé), génère un résumé hebdomadaire PROPRE.\n"
#     "Objectifs: dédoublonner, regrouper ce qui se répète, garder chiffres/échéances exacts, zéro hallucination, phrases complètes.\n"
#     "Style: fluide, cohérent, structuré. Ne pas tronquer.\n"
# )
# WEEKLY_USER_TMPL = """Contenu agrégé (par rubrique) :
# ---
# [[WEEKLY_INPUT]]
# ---
# Attendu (JSON strict) :
# {
#   "global_summary": "Quelques lignes par catégories. Ne pas omettre d'informations. Paragraphe structuré, fluide. Un retour à la ligne par thème (m&a, primary market, results, spreads, autres infos).",
#   "ma_and_ratings": ["2 à 8 puces propres"],
#   "results": ["2 à 8 puces propres"],
#   "financial_credit_spreads": ["2 à 8 puces propres"],
#   "primary_market": ["2 à 8 puces propres"],
#   "other_important_infos": ["2 à 8 puces propres"]
# }"""

# # ---- MONTHLY SYNTH PROMPT ----
# MONTHLY_SYS = (
#     "Tu es analyste crédit. À partir de weeklies, produis une synthèse mensuelle.\n"
#     "Objectifs: tendances du mois, thèmes dominants, événements majeurs, risques et pipeline/primaire.\n"
#     "Conserve chiffres/échéances exacts, zéro hallucination; phrases complètes.\n"
#     "Style: fluide, cohérent, structuré. Ne pas tronquer. Pas de redondance.\n"
# )
# MONTHLY_USER_TMPL = """Contenu agrégé weeklies (par rubrique) :
# ---
# [[MONTHLY_INPUT]]
# ---
# Attendu (JSON strict) :
# {
#   "global_summary": "6–16 puces synthétiques (thèmes, drivers, risques, highlights).",
#   "ma_and_ratings": ["2–10 puces propres"],
#   "results": ["2–10 puces propres"],
#   "financial_credit_spreads": ["2–10 puces propres"],
#   "primary_market": ["2–10 puces propres"],
#   "other_important_infos": ["2–10 puces propres"]
# }"""

# # --- helpers sortie LLM ---
# def _flatten_spaces(s: str) -> str:
#     return re.sub(r"\s*\n+\s*", " ", str(s or "")).strip()

# def _to_items(value) -> list[str]:
#     if isinstance(value, list):
#         return [_flatten_spaces(x).lstrip("•-–—* ").strip() for x in value if str(x).strip()]
#     lines = [ln.rstrip() for ln in str(value or "").splitlines() if ln.strip()]
#     items, cur = [], []
#     for ln in lines:
#         if re.match(r"^\s*[-•–—]\s+", ln):
#             if cur: items.append(_flatten_spaces(" ".join(cur)))
#             cur = [ln.lstrip().lstrip("-•–—* ").strip()]
#         else:
#             cur.append(ln.strip())
#     if cur: items.append(_flatten_spaces(" ".join(cur)))
#     items = [re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()]
#     return items

# def classify_with_openai(text: str) -> Dict[str, str]:
#     if not client:
#         return {k:"" for k in DAILY_FIELDS}
#     body = (text or "")[:80000]
#     usr  = DAILY_USER_TMPL.replace("[[EMAIL_BODY]]", body)
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type": "json_object"},
#             temperature=0.1,
#             messages=[{"role":"system","content":DAILY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (classify): {e}")
#         parsed = {}
#     cleaned = {}
#     for k in DAILY_FIELDS:
#         raw = parsed.get(k, "")
#         items = _to_items(raw)
#         cleaned[k] = "\n".join(items)
#     return cleaned

# def synth_llm_weekly(weekly_input_text: str) -> Dict[str,str]:
#     if not client:
#         return {"global_summary": summarize_texts([weekly_input_text], 12), **{k: "" for k in DAILY_FIELDS}}
#     usr = WEEKLY_USER_TMPL.replace("[[WEEKLY_INPUT]]", weekly_input_text[:80000])
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type":"json_object"},
#             temperature=0.2,
#             messages=[{"role":"system","content":WEEKLY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (weekly): {e}")
#         parsed = {}
#     out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
#     for k in DAILY_FIELDS:
#         out[k] = "\n".join(_to_items(parsed.get(k,"")))
#     return out

# def synth_llm_monthly(monthly_input_text: str) -> Dict[str,str]:
#     if not client:
#         return {"global_summary": summarize_texts([monthly_input_text], 16), **{k: "" for k in DAILY_FIELDS}}
#     usr = MONTHLY_USER_TMPL.replace("[[MONTHLY_INPUT]]", monthly_input_text[:120000])
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type":"json_object"},
#             temperature=0.2,
#             messages=[{"role":"system","content":MONTHLY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (monthly): {e}")
#         parsed = {}
#     out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
#     for k in DAILY_FIELDS:
#         out[k] = "\n".join(_to_items(parsed.get(k,"")))
#     return out

# # =========================
# # AGRÉGATION SOURCES
# # =========================
# def search_emails_in_range(start_dt: datetime, end_dt: datetime, subject_filter: str) -> List[dict]:
#     """Filtre les e-mails par fenêtre de dates + motif d’objet (case-insensitive)."""
#     results = []
#     for m in fetch_gmail_messages(newer_than_days=GMAIL_SEARCH_DAYS, limit=MAX_EMAILS_FETCH):
#         if not m.get("dt"): 
#             continue
#         if subject_filter.lower() not in (m["subject"] or "").lower():
#             continue
#         if not (start_dt <= m["dt"] <= end_dt):
#             continue
#         txt = strip_quotes_and_signatures(m["body"])
#         results.append({**m, "body": txt})
#     # tri chrono ascendant
#     results.sort(key=lambda x: x["dt"])
#     return results

# def collect_sources_for_range(
#     start_dt: datetime,
#     end_dt: datetime,
#     subject_filter: str,
#     uploaded_pdfs,
#     emails_override: List[dict] | None = None
# ) -> List[dict]:
#     """
#     Récupère:
#       - soit la liste d’e-mails fournie (emails_override),
#       - soit les e-mails IMAP filtrés par dates+objet,
#       - et les PDFs déposés.
#     Retourne [{kind:'email'|'pdf', dt, title, text}]
#     """
#     docs = []
#     # Emails (override si fourni)
#     emails = emails_override if emails_override is not None else search_emails_in_range(start_dt, end_dt, subject_filter)
#     for m in emails:
#         docs.append({"kind":"email","dt":m["dt"],"title":m["subject"],"text":m["body"]})

#     # PDFs
#     for f in (uploaded_pdfs or []):
#         content = f.read()
#         text = read_pdf_bytes_to_text(content)
#         dt_guess = guess_date_from_filename(f.name) or datetime.now()
#         docs.append({"kind":"pdf","dt":dt_guess,"title":f.name,"text":text})

#     docs.sort(key=lambda x: x["dt"])  # chrono ascendant (plus ancien en haut)
#     return docs

# def classify_docs_to_fields(docs: List[dict]) -> Dict[str, List[Tuple[datetime, str]]]:
#     """
#     Retourne un dict rubrique -> liste [(dt, item1), (dt, item2), ...], trié par dt asc.
#     """
#     per_field = {k: [] for k in DAILY_FIELDS}
#     for d in docs:
#         if not d["text"].strip():
#             continue
#         buckets = classify_with_openai(d["text"])
#         # pour chaque rubrique, séparer les items (1 item = 1 ligne), et attacher la date
#         for k in DAILY_FIELDS:
#             raw = buckets.get(k, "") or ""
#             for ln in [x.strip() for x in raw.splitlines() if x.strip()]:
#                 per_field[k].append((d["dt"], ln))
#     # ordonner chrono asc par rubrique
#     for k in DAILY_FIELDS:
#         per_field[k].sort(key=lambda t: t[0])
#     return per_field

# def join_fields_as_input(per_field: Dict[str, List[Tuple[datetime,str]]]) -> str:
#     """
#     Construit le bloc d’entrée pour la synthèse LLM, ordonné par thème puis chrono (asc).
#     """
#     blocks = []
#     for k in DAILY_FIELDS:
#         items = per_field.get(k, [])
#         if not items: 
#             continue
#         lines = [f"- {txt}" for _, txt in items]
#         blocks.append(f"## {k}\n" + "\n".join(lines))
#     return "\n\n".join(blocks)

# # =========================
# # UI
# # =========================
# st.sidebar.title("Menu")
# page = st.sidebar.radio("Catégories", ["Weekly","Monthly"], index=0)
# st.sidebar.caption("💾 JSON dans ./data/")

# st.title("🗞️ Hybrids Financial Debts Views — Weekly / Monthly")

# # ---- WEEKLY ----
# if page=="Weekly":
#     st.subheader("Weekly")

#     # Semaine (on choisit un jour -> on calcule lundi/dimanche)
#     chosen_day = st.date_input("Choisir un jour de la semaine", value=date.today(), format="YYYY-MM-DD")
#     monday, sunday = monday_sunday(chosen_day)
#     st.caption(f"Semaine: {monday.isoformat()} → {sunday.isoformat()}  (thèmes puis chrono du plus ancien au plus récent)")

#     # Browse e-mails (semaine)
#     start_dt = datetime.combine(monday, datetime.min.time())
#     end_dt   = datetime.combine(sunday, datetime.max.time())

#     colA, colB = st.columns([1,1])
#     with colA:
#         if st.button("🔍 Parcourir les e-mails (semaine)"):
#             emails = search_emails_in_range(start_dt, end_dt, SUBJECT_WEEKLY_FILTER)
#             st.session_state["weekly_email_list"] = emails
#             st.session_state["weekly_selected_ids"] = {m["message_id"] for m in emails}
#     with colB:
#         st.caption("Filtre objet: " + SUBJECT_WEEKLY_FILTER)

#     if "weekly_email_list" in st.session_state:
#         with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
#             sel = st.session_state.get("weekly_selected_ids", set())
#             new_sel = set(sel)
#             for m in st.session_state["weekly_email_list"]:
#                 mid = m["message_id"]
#                 checked = st.checkbox(
#                     f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
#                     key=f"wk_mail_{mid}",
#                     value=(mid in sel)
#                 )
#                 if checked: new_sel.add(mid)
#                 else:       new_sel.discard(mid)
#                 with st.expander("Voir l’aperçu", expanded=False):
#                     st.write((m["body"] or "")[:1500] + ("..." if len(m["body"])>1500 else ""))
#             st.session_state["weekly_selected_ids"] = new_sel

#     # Choix affichage rubriques
#     st.markdown("**Rubriques à afficher**")
#     cols = st.columns(5); chosen=[]
#     for i,f in enumerate(DAILY_FIELDS):
#         with cols[i%5]:
#             if st.checkbox(f.replace('_',' ').title(), value=True, key=f"w_{f}"): chosen.append(f)
#     chosen = chosen or DAILY_FIELDS

# # --- Upload PDFs (optionnel si tu utilises des emails) ---
# st.markdown("**Déposez des e-mails/notes au format PDF (optionnel si vous utilisez les e-mails)**")
# pdf_files = st.file_uploader("Glisser-déposer ici", type=["pdf"], accept_multiple_files=True)

# # --- Générer ---
# if st.button("⚙️ Générer le Weekly"):
#     # emails sélectionnés (si l’utilisateur a cliqué sur "Parcourir les e-mails")
#     selected_emails = []
#     if "weekly_email_list" in st.session_state:
#         ids = st.session_state.get("weekly_selected_ids", set())
#         selected_emails = [m for m in st.session_state["weekly_email_list"] if m["message_id"] in ids]

#     pdf_cnt = len(pdf_files or [])

#     # Règle : on peut générer si (au moins 1 e-mail sélectionné) OU (≥ 5 PDFs).
#     if not selected_emails and pdf_cnt < 5:
#         st.error("Sélectionnez au moins **un e-mail** OU déposez **≥ 5 PDFs** pour générer le Weekly.")
#     else:
#         # Fenêtre temporelle de la semaine
#         start_dt = datetime.combine(monday, datetime.min.time())
#         end_dt   = datetime.combine(sunday, datetime.max.time())

#         # Si on a des emails sélectionnés, on les passe en override. Sinon on n’utilise que les PDFs.
#         emails_override = selected_emails if selected_emails else []

#         docs = collect_sources_for_range(
#             start_dt, end_dt,
#             SUBJECT_WEEKLY_FILTER,
#             pdf_files,
#             emails_override=emails_override
#         )

#         if not docs:
#             st.warning("Aucune source sélectionnée/trouvée (vérifie la sélection d’e-mails ou tes PDFs).")
#         else:
#             per_field = classify_docs_to_fields(docs)
#             weekly_input = join_fields_as_input(per_field)
#             llm = synth_llm_weekly(weekly_input)

#             fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
#             global_summary = dedupe_bullets(llm.get("global_summary",""))
#             week_key = iso_week_key(monday)

#             payload = {
#                 "week": week_key,
#                 "window": [monday.isoformat(), sunday.isoformat()],
#                 "fields": fields,
#                 "global_summary": global_summary,
#                 "sources": [{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
#             }
#             st.session_state["weekly_preview"] = payload
#             st.success("Weekly généré.")

#     if "weekly_preview" in st.session_state:
#         wk = st.session_state["weekly_preview"]
#         st.markdown(f"### Weekly : {wk['week']}")
#         st.write("Fenêtre :", " → ".join(wk.get("window", [])) or "N/A")
#         with st.expander("Résumé global", expanded=True):
#             st.markdown(md_block(wk.get("global_summary","")))
#         for k in DAILY_FIELDS:
#             if k in chosen:
#                 with st.expander(k.replace("_"," ").title(), expanded=False):
#                     st.markdown(md_block(wk["fields"].get(k,"")))
#         with st.expander("Sources utilisées"):
#             for s in wk.get("sources", []):
#                 st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
#         if st.button("💾 Enregistrer ce Weekly"):
#             save_json(weekly_path(wk['week']), wk); st.success(f"Weekly sauvegardé : {weekly_path(wk['week'])}")

# # ---- MONTHLY ----
# elif page=="Monthly":
#     st.subheader("Monthly")

#     today = date.today()
#     y = st.number_input("Année", value=today.year, step=1)
#     m = st.number_input("Mois", 1, 12, value=today.month, step=1)
#     first = date(int(y), int(m), 1)
#     nxt   = first + relativedelta(months=1)
#     st.caption(f"Période: {first.isoformat()} → {(nxt - timedelta(days=1)).isoformat()} (thèmes puis chrono du plus ancien au plus récent)")

#     # Browse e-mails (mois)
#     start_dt = datetime.combine(first, datetime.min.time())
#     end_dt   = datetime.combine(nxt - timedelta(seconds=1), datetime.max.time())

#     colA, colB = st.columns([1,1])
#     with colA:
#         if st.button("🔍 Parcourir les e-mails (mois)"):
#             emails = search_emails_in_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER)
#             st.session_state["monthly_email_list"] = emails
#             st.session_state["monthly_selected_ids"] = {m["message_id"] for m in emails}
#     with colB:
#         st.caption("Filtre objet: " + SUBJECT_MONTHLY_FILTER)

#     if "monthly_email_list" in st.session_state:
#         with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
#             sel = st.session_state.get("monthly_selected_ids", set())
#             new_sel = set(sel)
#             for m in st.session_state["monthly_email_list"]:
#                 mid = m["message_id"]
#                 checked = st.checkbox(
#                     f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
#                     key=f"mo_mail_{mid}",
#                     value=(mid in sel)
#                 )
#                 if checked: new_sel.add(mid)
#                 else:       new_sel.discard(mid)
#                 with st.expander("Voir l’aperçu", expanded=False):
#                     st.write((m["body"] or "")[:5000] + ("..." if len(m["body"])>5000 else ""))
#             st.session_state["monthly_selected_ids"] = new_sel

#     # Rubriques à afficher
#     st.markdown("**Rubriques à afficher**")
#     cols = st.columns(5); chosen=[]
#     for i,f in enumerate(DAILY_FIELDS):
#         with cols[i%5]:
#             if st.checkbox(f.replace('_',' ').title(), value=True, key=f"m_{f}"): chosen.append(f)
#     chosen = chosen or DAILY_FIELDS

#     # Upload PDFs (optionnel)
#     st.markdown("**(Optionnel) Déposer des PDF**")
#     pdf_files = st.file_uploader("Glisser-déposer des PDF (facultatif)", type=["pdf"], accept_multiple_files=True, key="monthly_pdf")

#     if st.button("⚙️ Générer le Monthly"):
#         emails_override = None
#         if "monthly_email_list" in st.session_state:
#             ids = st.session_state.get("monthly_selected_ids", set())
#             emails_override = [m for m in st.session_state["monthly_email_list"] if m["message_id"] in ids]

#         docs = collect_sources_for_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER, pdf_files, emails_override=emails_override)
#         if not docs:
#             st.warning("Aucune source sélectionnée/trouvée.")
#         else:
#             per_field = classify_docs_to_fields(docs)
#             monthly_input = join_fields_as_input(per_field)
#             llm = synth_llm_monthly(monthly_input)
#             fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
#             global_summary = dedupe_bullets(llm.get("global_summary",""))
#             month_key = f"{int(y):04d}-{int(m):02d}"
#             payload = {
#                 "month": month_key,
#                 "fields": fields,
#                 "global_summary": global_summary,
#                 "sources":[{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
#             }
#             st.session_state["monthly_preview"]=payload
#             st.success("Monthly généré.")

#     if "monthly_preview" in st.session_state:
#         mo = st.session_state["monthly_preview"]
#         st.markdown(f"### Monthly : {mo['month']}")
#         with st.expander("Résumé global", expanded=True):
#             st.markdown(md_block(mo.get("global_summary","")))
#         for k in DAILY_FIELDS:
#             if k in chosen:
#                 with st.expander(k.replace("_"," ").title(), expanded=False):
#                     st.markdown(md_block(mo["fields"].get(k,"")))
#         with st.expander("Sources utilisées"):
#             for s in mo.get("sources", []):
#                 st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
#         if st.button("💾 Enregistrer ce Monthly"):
#             save_json(monthly_path(mo['month']), mo); st.success(f"Monthly sauvegardé : {monthly_path(mo['month'])}")



# from __future__ import annotations
# import os, re, json, imaplib, email, io
# from email.header import decode_header, make_header
# from email.utils import parsedate_to_datetime
# from pathlib import Path
# from datetime import date, datetime, timedelta
# from typing import Dict, List, Tuple

# import streamlit as st
# from dateutil.relativedelta import relativedelta

# # =========================
# # GATE (facultatif)
# # =========================
# gate = st.secrets.get("APP_PASSWORD", "")
# if gate:
#     pw = st.text_input("Mot de passe", type="password")
#     if pw != gate:
#         st.stop()

# # =========================
# # CONFIG
# # =========================
# st.set_page_config(page_title="Weekly / Monthly – Hybrids Views", layout="wide")

# # ⚠️ Lire les secrets depuis st.secrets (ne pas hardcoder)
# GMAIL_ADDRESS       = st.secrets.get("GMAIL_ADDRESS", "")
# GMAIL_APP_PASSWORD  = st.secrets.get("GMAIL_APP_PASSWORD", "")
# OPENAI_API_KEY      = st.secrets.get("OPENAI_API_KEY", "")
# OPENAI_MODEL        = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

# SUBJECT_WEEKLY_FILTER  = st.secrets.get("SUBJECT_WEEKLY_FILTER",  "Hybrids Financial Debts Views")
# SUBJECT_MONTHLY_FILTER = st.secrets.get("SUBJECT_MONTHLY_FILTER", "Hybrids Financial Debts Views Weekly")

# GMAIL_SEARCH_DAYS = int(st.secrets.get("GMAIL_SEARCH_DAYS", 30))
# MAX_EMAILS_FETCH  = int(st.secrets.get("MAX_EMAILS_FETCH", 400))

# DATA_DIR    = Path("data")
# WEEKLY_DIR  = DATA_DIR / "weekly"
# MONTHLY_DIR = DATA_DIR / "monthly"
# for d in (DATA_DIR, WEEKLY_DIR, MONTHLY_DIR): d.mkdir(parents=True, exist_ok=True)

# # Rubriques (ordre thématique)
# DAILY_FIELDS = ["ma_and_ratings","results","financial_credit_spreads","primary_market","other_important_infos"]

# # =========================
# # HELPERS FS/FORMAT
# # =========================
# def weekly_path(w: str) -> Path:   return WEEKLY_DIR / f"{w}.json"
# def monthly_path(m: str) -> Path:  return MONTHLY_DIR / f"{m}.json"

# def load_json(p: Path, default):
#     if p.exists():
#         try: return json.loads(p.read_text(encoding="utf-8"))
#         except: return default
#     return default

# def save_json(p: Path, payload): p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

# def iso_week_key(d: date) -> str:
#     y, w, _ = d.isocalendar()
#     return f"{y}-W{w:02d}"

# def monday_sunday(d: date) -> Tuple[date, date]:
#     monday = d - timedelta(days=d.weekday())
#     sunday = monday + timedelta(days=6)
#     return monday, sunday

# def md_block(text: str) -> str:
#     return (text or "").strip().replace("\n", "  \n") or "_(vide)_"

# def normalize_line(s: str) -> str:
#     return re.sub(r"\s+", " ", s.strip().strip("•-–·* ")).lower()

# def dedupe_bullets(text: str) -> str:
#     seen, out = set(), []
#     for ln in (text or "").splitlines():
#         base = normalize_line(ln)
#         if base and base not in seen:
#             seen.add(base)
#             out.append(f"- {ln.strip().strip('•-–·* ')}")
#     return "\n".join(out)

# def summarize_texts(texts: List[str], max_bullets=12) -> str:
#     tokens = []
#     for t in texts or []:
#         for ln in str(t or "").splitlines():
#             s = ln.strip(" •-–\t ")
#             if s: tokens.append(s)
#     uniq = list(dict.fromkeys(tokens))[:max_bullets]
#     return "\n".join(f"- {b}" for b in uniq)

# # =========================
# # GMAIL
# # =========================
# def _decode_header(s) -> str:
#     try: return str(make_header(decode_header(s or "")))
#     except: return s or ""

# def _strip_html(html: str) -> str:
#     t = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
#     t = re.sub(r"(?s)<.*?>", " ", t)
#     t = re.sub(r"[ \t]+", " ", t)
#     t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
#     return t.strip()

# def _message_to_text(msg: email.message.Message) -> str:
#     if msg.is_multipart():
#         for part in msg.walk():
#             if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
#                 try: return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
#                 except: pass
#         for part in msg.walk():
#             if part.get_content_type() == "text/html" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
#                 try: return _strip_html(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
#                 except: pass
#         payloads=[]
#         for part in msg.walk():
#             if part.get_content_maintype()=="text":
#                 try: payloads.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
#                 except: pass
#         return "\n\n".join(payloads)
#     else:
#         try:
#             payload = msg.get_payload(decode=True)
#             if payload is None: return str(msg.get_payload())
#             text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
#             return _strip_html(text) if msg.get_content_type()=="text/html" else text
#         except: return str(msg.get_payload())

# def fetch_gmail_messages(newer_than_days=30, limit=400) -> List[dict]:
#     res=[]
#     if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
#         return res
#     try:
#         imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
#         imap.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
#         ok,_ = imap.select("INBOX")
#         if ok!="OK": raise RuntimeError("INBOX non sélectionnable")
#         since = (datetime.utcnow() - timedelta(days=newer_than_days)).strftime("%d-%b-%Y")
#         ok, data = imap.search(None, f'(SINCE {since})')
#         if ok!="OK": return res
#         ids = (data[0].split() or [])[-limit:]
#         for msg_id in reversed(ids):
#             ok, msg_data = imap.fetch(msg_id, "(RFC822)")
#             if ok!="OK": continue
#             msg  = email.message_from_bytes(msg_data[0][1])
#             subj = _decode_header(msg.get("Subject",""))
#             body = _message_to_text(msg)
#             try:
#                 dt = parsedate_to_datetime(msg.get("Date"))
#                 if dt and dt.tzinfo: dt = dt.astimezone().replace(tzinfo=None)
#             except:
#                 dt = datetime.now()
#             mid = (msg.get("Message-Id") or msg.get("Message-ID") or "").strip() or f"INBOX:{msg_id.decode()}"
#             res.append({"subject": subj, "body": body, "dt": dt, "message_id": mid})
#         imap.logout()
#     except Exception as e:
#         st.error(f"Erreur Gmail IMAP: {e}")
#     return res

# # =========================
# # CLEAN BODY (quotes/signatures)
# # =========================
# QUOTE_PATTERNS = [
#     r"^On .+ wrote:$", r"^Le .+ a écrit :", r"^From: .+$", r"^-{2,}\s*Original Message\s*-{2,}$",
#     r"^> .*", r"^__+$", r"^\s*De : .+$", r"^\s*Envoyé : .+$"
# ]
# def strip_quotes_and_signatures(txt: str) -> str:
#     lines, out = (txt or "").splitlines(), []
#     for ln in lines:
#         if any(re.match(p, ln.strip(), re.IGNORECASE) for p in QUOTE_PATTERNS): break
#         out.append(ln)
#     joined = "\n".join(out)
#     joined = re.sub(r"(?is)this message.*?confidential.*", "", joined)
#     joined = re.sub(r"(?is)ce message.*?confidentiel.*", "", joined)
#     return joined.strip()

# # =========================
# # PDF LOADER
# # =========================
# def read_pdf_bytes_to_text(data: bytes) -> str:
#     try:
#         import PyPDF2
#         reader = PyPDF2.PdfReader(io.BytesIO(data))
#         pages = []
#         for p in reader.pages:
#             try:
#                 pages.append(p.extract_text() or "")
#             except:
#                 pages.append("")
#         return "\n".join(pages)
#     except Exception as e:
#         st.error(f"Lecture PDF: {e}")
#         return ""

# def guess_date_from_filename(name: str) -> datetime|None:
#     m = re.search(r"(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])", name)
#     if not m: return None
#     try:
#         s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
#         return datetime.fromisoformat(s)
#     except:
#         return None

# # =========================
# # OPENAI (Classifier + Synthèses)
# # =========================
# from openai import OpenAI
# client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# DAILY_SYS = (
#     "Tu es un assistant financier. Classe TOUT le contenu pertinent dans EXACTEMENT ces 5 catégories.\n"
#     "Aucune paraphrase. Recopie le texte source tel quel, en phrases complètes (jamais tronquées).\n"
#     "1 item = 1 phrase autonome (pas de virgule finale/parenthèse ouverte).\n"
#     "Si des retours à la ligne coupent une phrase, RECOLLE les segments.\n"
#     "IMPORTANT : conserve l’ORDRE CHRONOLOGIQUE d’apparition du texte source.\n"
#     "IMPORTANT : préfixe chaque item par la date au format [YYYY-MM-DD] si elle est fournie.\n"
#     "Exclure signatures/politesse/disclaimers.\n"
#     "En français."
# )
# DAILY_USER_TMPL = """Texte :
# ---
# [[EMAIL_BODY]]
# ---
# Retourne un JSON strict:
# {
#   "financial_credit_spreads": ["..."],
#   "primary_market": ["..."],
#   "results": ["..."],
#   "ma_and_ratings": ["..."],
#   "other_important_infos": ["..."]
# }"""

# WEEKLY_SYS = (
#     "Tu es analyste crédit. À partir d’un contenu agrégé (déjà classé), génère un résumé hebdomadaire PROPRE.\n"
#     "Objectifs: dédoublonner, regrouper ce qui se répète, garder chiffres/échéances exacts, zéro hallucination, phrases complètes.\n"
#     "Style: fluide, cohérent, structuré. Ne pas tronquer.\n"
# )
# WEEKLY_USER_TMPL = """Contenu agrégé (par rubrique) :
# ---
# [[WEEKLY_INPUT]]
# ---
# Attendu (JSON strict) :
# {
#   "global_summary": "Quelques lignes par catégories. Ne pas omettre d'informations. Paragraphe structuré, fluide. Un retour à la ligne par thème (m&a, primary market, results, spreads, autres infos).",
#   "ma_and_ratings": ["2 à 8 puces propres"],
#   "results": ["2 à 8 puces propres"],
#   "financial_credit_spreads": ["2 à 8 puces propres"],
#   "primary_market": ["2 à 8 puces propres"],
#   "other_important_infos": ["2 à 8 puces propres"]
# }"""

# MONTHLY_SYS = (
#     "Tu es analyste crédit. À partir de weeklies, produis une synthèse mensuelle.\n"
#     "Objectifs: tendances du mois, thèmes dominants, événements majeurs, risques et pipeline/primaire.\n"
#     "Conserve chiffres/échéances exacts, zéro hallucination; phrases complètes.\n"
#     "Style: fluide, cohérent, structuré. Ne pas tronquer. Pas de redondance.\n"
# )
# MONTHLY_USER_TMPL = """Contenu agrégé weeklies (par rubrique) :
# ---
# [[MONTHLY_INPUT]]
# ---
# Attendu (JSON strict) :
# {
#   "global_summary": "6–16 puces synthétiques (thèmes, drivers, risques, highlights).",
#   "ma_and_ratings": ["2–10 puces propres"],
#   "results": ["2–10 puces propres"],
#   "financial_credit_spreads": ["2–10 puces propres"],
#   "primary_market": ["2–10 puces propres"],
#   "other_important_infos": ["2–10 puces propres"]
# }"""

# def _flatten_spaces(s: str) -> str:
#     return re.sub(r"\s*\n+\s*", " ", str(s or "")).strip()

# def _to_items(value) -> list[str]:
#     if isinstance(value, list):
#         return [_flatten_spaces(x).lstrip("•-–—* ").strip() for x in value if str(x).strip()]
#     lines = [ln.rstrip() for ln in str(value or "").splitlines() if ln.strip()]
#     items, cur = [], []
#     for ln in lines:
#         if re.match(r"^\s*[-•–—]\s+", ln):
#             if cur: items.append(_flatten_spaces(" ".join(cur)))
#             cur = [ln.lstrip().lstrip("-•–—* ").strip()]
#         else:
#             cur.append(ln.strip())
#     if cur: items.append(_flatten_spaces(" ".join(cur)))
#     items = [re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()]
#     return items

# def classify_with_openai(text: str) -> Dict[str, str]:
#     if not client:
#         return {k:"" for k in DAILY_FIELDS}
#     body = (text or "")[:80000]
#     usr  = DAILY_USER_TMPL.replace("[[EMAIL_BODY]]", body)
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type": "json_object"},
#             temperature=0.1,
#             messages=[{"role":"system","content":DAILY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (classify): {e}")
#         parsed = {}
#     cleaned = {}
#     for k in DAILY_FIELDS:
#         raw = parsed.get(k, "")
#         items = _to_items(raw)
#         cleaned[k] = "\n".join(items)
#     return cleaned

# def synth_llm_weekly(weekly_input_text: str) -> Dict[str,str]:
#     if not client:
#         return {"global_summary": summarize_texts([weekly_input_text], 12), **{k: "" for k in DAILY_FIELDS}}
#     usr = WEEKLY_USER_TMPL.replace("[[WEEKLY_INPUT]]", weekly_input_text[:80000])
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type":"json_object"},
#             temperature=0.2,
#             messages=[{"role":"system","content":WEEKLY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (weekly): {e}")
#         parsed = {}
#     out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
#     for k in DAILY_FIELDS:
#         out[k] = "\n".join(_to_items(parsed.get(k,"")))
#     return out

# def synth_llm_monthly(monthly_input_text: str) -> Dict[str,str]:
#     if not client:
#         return {"global_summary": summarize_texts([monthly_input_text], 16), **{k: "" for k in DAILY_FIELDS}}
#     usr = MONTHLY_USER_TMPL.replace("[[MONTHLY_INPUT]]", monthly_input_text[:120000])
#     try:
#         r = client.chat.completions.create(
#             model=OPENAI_MODEL,
#             response_format={"type":"json_object"},
#             temperature=0.2,
#             messages=[{"role":"system","content":MONTHLY_SYS},{"role":"user","content":usr}],
#         )
#         parsed = json.loads(r.choices[0].message.content or "{}")
#     except Exception as e:
#         st.error(f"OpenAI error (monthly): {e}")
#         parsed = {}
#     out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
#     for k in DAILY_FIELDS:
#         out[k] = "\n".join(_to_items(parsed.get(k,"")))
#     return out

# # =========================
# # ONE-SHOT CLASSIFY HELPERS
# # =========================
# DATE_TAG_RE = re.compile(r"^\s*\[([0-9]{4}-[0-9]{2}-[0-9]{2})\]\s*(.*)")

# def build_aggregate_text(docs: List[dict]) -> str:
#     chunks = []
#     for d in sorted(docs, key=lambda x: x["dt"]):
#         ddate = d["dt"].date().isoformat()
#         title = (d.get("title") or "").strip()
#         text  = (d.get("text")  or "").strip()
#         if not text:
#             continue
#         head = f"[{ddate}] {title}".strip()
#         chunks.append(f"{head}\n{text}")
#     return "\n\n---\n\n".join(chunks)

# def ensure_dash_bullets(s: str) -> str:
#     return "\n".join(f"- {ln.strip().lstrip('- ')}" for ln in (s or "").splitlines() if ln.strip())

# def sort_items_by_date_prefix(text_block: str) -> str:
#     rows = [ln.strip().lstrip("- ").strip() for ln in (text_block or "").splitlines() if ln.strip()]
#     tagged = []
#     for i, row in enumerate(rows):
#         m = DATE_TAG_RE.match(row)
#         if m:
#             try:
#                 dt = datetime.fromisoformat(m.group(1)).date()
#             except:
#                 dt = date.min
#             txt = m.group(2).strip()
#             tagged.append((dt, i, txt))
#         else:
#             tagged.append((date.min, i, row))
#     tagged.sort(key=lambda t: (t[0], t[1]))
#     return "\n".join(f"- {t[2]}" for t in tagged)

# def classify_all_at_once(docs: List[dict]) -> Dict[str, str]:
#     big_text = build_aggregate_text(docs)
#     buckets  = classify_with_openai(big_text)  # UN seul appel
#     out = {}
#     for k in DAILY_FIELDS:
#         value = ensure_dash_bullets(buckets.get(k, ""))
#         value = dedupe_bullets(value)
#         value = sort_items_by_date_prefix(value)
#         out[k] = value
#     return out

# # =========================
# # AGRÉGATION SOURCES
# # =========================
# def search_emails_in_range(start_dt: datetime, end_dt: datetime, subject_filter: str) -> List[dict]:
#     results = []
#     for m in fetch_gmail_messages(newer_than_days=GMAIL_SEARCH_DAYS, limit=MAX_EMAILS_FETCH):
#         if not m.get("dt"): 
#             continue
#         if subject_filter.lower() not in (m["subject"] or "").lower():
#             continue
#         if not (start_dt <= m["dt"] <= end_dt):
#             continue
#         txt = strip_quotes_and_signatures(m["body"])
#         results.append({**m, "body": txt})
#     results.sort(key=lambda x: x["dt"])
#     return results

# def collect_sources_for_range(
#     start_dt: datetime,
#     end_dt: datetime,
#     subject_filter: str,
#     uploaded_pdfs,
#     emails_override: List[dict] | None = None
# ) -> List[dict]:
#     docs = []
#     emails = emails_override if emails_override is not None else search_emails_in_range(start_dt, end_dt, subject_filter)
#     for m in emails:
#         docs.append({"kind":"email","dt":m["dt"],"title":m["subject"],"text":m["body"]})
#     for f in (uploaded_pdfs or []):
#         content = f.read()
#         text = read_pdf_bytes_to_text(content)
#         dt_guess = guess_date_from_filename(f.name) or datetime.now()
#         docs.append({"kind":"pdf","dt":dt_guess,"title":f.name,"text":text})
#     docs.sort(key=lambda x: x["dt"])
#     return docs

# # =========================
# # UI
# # =========================
# st.sidebar.title("Menu")
# page = st.sidebar.radio("Catégories", ["Weekly","Monthly"], index=0)
# st.sidebar.caption("💾 JSON dans ./data/")

# st.title("🗞️ Hybrids Financial Debts Views — Weekly / Monthly")

# # ---- WEEKLY ----
# if page=="Weekly":
#     st.subheader("Weekly")

#     chosen_day = st.date_input("Choisir un jour de la semaine", value=date.today(), format="YYYY-MM-DD")
#     monday, sunday = monday_sunday(chosen_day)
#     st.caption(f"Semaine: {monday.isoformat()} → {sunday.isoformat()} (thèmes puis chrono du plus ancien au plus récent)")

#     # Browse e-mails (semaine)
#     start_dt = datetime.combine(monday, datetime.min.time())
#     end_dt   = datetime.combine(sunday, datetime.max.time())

#     colA, colB = st.columns([1,1])
#     with colA:
#         if st.button("🔍 Parcourir les e-mails (semaine)"):
#             emails = search_emails_in_range(start_dt, end_dt, SUBJECT_WEEKLY_FILTER)
#             st.session_state["weekly_email_list"] = emails
#             st.session_state["weekly_selected_ids"] = {m["message_id"] for m in emails}
#     with colB:
#         st.caption("Filtre objet: " + SUBJECT_WEEKLY_FILTER)

#     if "weekly_email_list" in st.session_state:
#         with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
#             sel = st.session_state.get("weekly_selected_ids", set())
#             new_sel = set(sel)
#             for m in st.session_state["weekly_email_list"]:
#                 mid = m["message_id"]
#                 checked = st.checkbox(
#                     f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
#                     key=f"wk_mail_{mid}",
#                     value=(mid in sel)
#                 )
#                 if checked: new_sel.add(mid)
#                 else:       new_sel.discard(mid)
#                 with st.expander("Voir l’aperçu", expanded=False):
#                     st.write((m["body"] or "")[:1500] + ("..." if len(m["body"])>1500 else ""))
#             st.session_state["weekly_selected_ids"] = new_sel

#     # Choix affichage rubriques
#     st.markdown("**Rubriques à afficher**")
#     cols = st.columns(5); chosen=[]
#     for i,f in enumerate(DAILY_FIELDS):
#         with cols[i%5]:
#             if st.checkbox(f.replace('_',' ').title(), value=True, key=f"w_{f}"): chosen.append(f)
#     chosen = chosen or DAILY_FIELDS

#     # PDFs optionnels (si pas d'emails, exiger ≥5)
#     st.markdown("**Déposez des e-mails/notes au format PDF (optionnel si vous utilisez les e-mails)**")
#     pdf_files = st.file_uploader("Glisser-déposer ici", type=["pdf"], accept_multiple_files=True)

#     if st.button("⚙️ Générer le Weekly"):
#         selected_emails = []
#         if "weekly_email_list" in st.session_state:
#             ids = st.session_state.get("weekly_selected_ids", set())
#             selected_emails = [m for m in st.session_state["weekly_email_list"] if m["message_id"] in ids]

#         pdf_cnt = len(pdf_files or [])
#         if not selected_emails and pdf_cnt < 5:
#             st.error("Sélectionnez au moins **un e-mail** OU déposez **≥ 5 PDFs** pour générer le Weekly.")
#         else:
#             emails_override = selected_emails if selected_emails else []
#             docs = collect_sources_for_range(start_dt, end_dt, SUBJECT_WEEKLY_FILTER, pdf_files, emails_override=emails_override)
#             if not docs:
#                 st.warning("Aucune source sélectionnée/trouvée (vérifiez e-mails sélectionnés ou PDFs).")
#             else:
#                 # --- NOUVEAU : 1 seul appel de tri ---
#                 categorized = classify_all_at_once(docs)
#                 weekly_input = "\n\n".join(
#                     f"## {k}\n{categorized[k]}" for k in DAILY_FIELDS if categorized.get(k, "").strip()
#                 )
#                 # 1 appel de synthèse
#                 llm = synth_llm_weekly(weekly_input)
#                 fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
#                 global_summary = dedupe_bullets(llm.get("global_summary",""))
#                 week_key = iso_week_key(monday)

#                 payload = {
#                     "week": week_key,
#                     "window": [monday.isoformat(), sunday.isoformat()],
#                     "fields": fields,
#                     "global_summary": global_summary,
#                     "sources": [{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
#                 }
#                 st.session_state["weekly_preview"] = payload
#                 st.success("Weekly généré.")

#     if "weekly_preview" in st.session_state:
#         wk = st.session_state["weekly_preview"]
#         st.markdown(f"### Weekly : {wk['week']}")
#         st.write("Fenêtre :", " → ".join(wk.get("window", [])) or "N/A")
#         with st.expander("Résumé global", expanded=True):
#             st.markdown(md_block(wk.get("global_summary","")))
#         for k in DAILY_FIELDS:
#             if k in chosen:
#                 with st.expander(k.replace("_"," ").title(), expanded=False):
#                     st.markdown(md_block(wk["fields"].get(k,"")))
#         with st.expander("Sources utilisées"):
#             for s in wk.get("sources", []):
#                 st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
#         if st.button("💾 Enregistrer ce Weekly"):
#             save_json(weekly_path(wk['week']), wk); st.success(f"Weekly sauvegardé : {weekly_path(wk['week'])}")

# # ---- MONTHLY ----
# elif page=="Monthly":
#     st.subheader("Monthly")

#     today = date.today()
#     y = st.number_input("Année", value=today.year, step=1)
#     m = st.number_input("Mois", 1, 12, value=today.month, step=1)
#     first = date(int(y), int(m), 1)
#     nxt   = first + relativedelta(months=1)
#     st.caption(f"Période: {first.isoformat()} → {(nxt - timedelta(days=1)).isoformat()} (thèmes puis chrono du plus ancien au plus récent)")

#     # Browse e-mails (mois)
#     start_dt = datetime.combine(first, datetime.min.time())
#     end_dt   = datetime.combine(nxt - timedelta(seconds=1), datetime.max.time())

#     colA, colB = st.columns([1,1])
#     with colA:
#         if st.button("🔍 Parcourir les e-mails (mois)"):
#             emails = search_emails_in_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER)
#             st.session_state["monthly_email_list"] = emails
#             st.session_state["monthly_selected_ids"] = {m["message_id"] for m in emails}
#     with colB:
#         st.caption("Filtre objet: " + SUBJECT_MONTHLY_FILTER)

#     if "monthly_email_list" in st.session_state:
#         with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
#             sel = st.session_state.get("monthly_selected_ids", set())
#             new_sel = set(sel)
#             for m in st.session_state["monthly_email_list"]:
#                 mid = m["message_id"]
#                 checked = st.checkbox(
#                     f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
#                     key=f"mo_mail_{mid}",
#                     value=(mid in sel)
#                 )
#                 if checked: new_sel.add(mid)
#                 else:       new_sel.discard(mid)
#                 with st.expander("Voir l’aperçu", expanded=False):
#                     st.write((m["body"] or "")[:1500] + ("..." if len(m["body"])>1500 else ""))
#             st.session_state["monthly_selected_ids"] = new_sel

#     # Rubriques à afficher
#     st.markdown("**Rubriques à afficher**")
#     cols = st.columns(5); chosen=[]
#     for i,f in enumerate(DAILY_FIELDS):
#         with cols[i%5]:
#             if st.checkbox(f.replace('_',' ').title(), value=True, key=f"m_{f}"): chosen.append(f)
#     chosen = chosen or DAILY_FIELDS

#     # PDFs optionnels
#     st.markdown("**(Optionnel) Déposer des PDF**")
#     pdf_files = st.file_uploader("Glisser-déposer des PDF (facultatif)", type=["pdf"], accept_multiple_files=True, key="monthly_pdf")

#     if st.button("⚙️ Générer le Monthly"):
#         emails_override = None
#         if "monthly_email_list" in st.session_state:
#             ids = st.session_state.get("monthly_selected_ids", set())
#             emails_override = [m for m in st.session_state["monthly_email_list"] if m["message_id"] in ids]

#         docs = collect_sources_for_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER, pdf_files, emails_override=emails_override)
#         if not docs:
#             st.warning("Aucune source sélectionnée/trouvée.")
#         else:
#             # 1 seul appel de tri
#             categorized = classify_all_at_once(docs)
#             monthly_input = "\n\n".join(
#                 f"## {k}\n{categorized[k]}" for k in DAILY_FIELDS if categorized.get(k, "").strip()
#             )
#             # 1 appel de synthèse mensuelle
#             llm = synth_llm_monthly(monthly_input)
#             fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
#             global_summary = dedupe_bullets(llm.get("global_summary",""))
#             month_key = f"{int(y):04d}-{int(m):02d}"
#             payload = {
#                 "month": month_key,
#                 "fields": fields,
#                 "global_summary": global_summary,
#                 "sources":[{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
#             }
#             st.session_state["monthly_preview"]=payload
#             st.success("Monthly généré.")

#     if "monthly_preview" in st.session_state:
#         mo = st.session_state["monthly_preview"]
#         st.markdown(f"### Monthly : {mo['month']}")
#         with st.expander("Résumé global", expanded=True):
#             st.markdown(md_block(mo.get("global_summary","")))
#         for k in DAILY_FIELDS:
#             if k in chosen:
#                 with st.expander(k.replace("_"," ").title(), expanded=False):
#                     st.markdown(md_block(mo["fields"].get(k,"")))
#         with st.expander("Sources utilisées"):
#             for s in mo.get("sources", []):
#                 st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
#         if st.button("💾 Enregistrer ce Monthly"):
#             save_json(monthly_path(mo['month']), mo); st.success(f"Monthly sauvegardé : {monthly_path(mo['month'])}")


from __future__ import annotations
import os, re, json, imaplib, email, io
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import streamlit as st
from dateutil.relativedelta import relativedelta

# =========================
# GATE (facultatif)
# =========================
gate = st.secrets.get("APP_PASSWORD", "")
if gate:
    pw = st.text_input("Mot de passe", type="password")
    if pw != gate:
        st.stop()

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Weekly / Monthly – Hybrids Views", layout="wide")

# ⚠️ Lire les secrets depuis st.secrets (ne pas hardcoder)
GMAIL_ADDRESS       = st.secrets.get("GMAIL_ADDRESS", "")
GMAIL_APP_PASSWORD  = st.secrets.get("GMAIL_APP_PASSWORD", "")
OPENAI_API_KEY      = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL        = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")

SUBJECT_WEEKLY_FILTER  = st.secrets.get("SUBJECT_WEEKLY_FILTER",  "Hybrids Financial Debts Views")
SUBJECT_MONTHLY_FILTER = st.secrets.get("SUBJECT_MONTHLY_FILTER", "Hybrids Financial Debts Views Weekly")

GMAIL_SEARCH_DAYS = int(st.secrets.get("GMAIL_SEARCH_DAYS", 30))
MAX_EMAILS_FETCH  = int(st.secrets.get("MAX_EMAILS_FETCH", 400))

DATA_DIR    = Path("data")
WEEKLY_DIR  = DATA_DIR / "weekly"
MONTHLY_DIR = DATA_DIR / "monthly"
for d in (DATA_DIR, WEEKLY_DIR, MONTHLY_DIR): d.mkdir(parents=True, exist_ok=True)

# Rubriques (ordre thématique)
DAILY_FIELDS = ["ma_and_ratings","results","financial_credit_spreads","primary_market","other_important_infos"]

# =========================
# HELPERS FS/FORMAT
# =========================
def weekly_path(w: str) -> Path:   return WEEKLY_DIR / f"{w}.json"
def monthly_path(m: str) -> Path:  return MONTHLY_DIR / f"{m}.json"

def load_json(p: Path, default):
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except: return default
    return default

def save_json(p: Path, payload): p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def iso_week_key(d: date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"

def monday_sunday(d: date) -> Tuple[date, date]:
    monday = d - timedelta(days=d.weekday())
    sunday = monday + timedelta(days=6)
    return monday, sunday

def md_block(text: str) -> str:
    return (text or "").strip().replace("\n", "  \n") or "_(vide)_"

def normalize_line(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().strip("•-–·* ")).lower()

def dedupe_bullets(text: str) -> str:
    seen, out = set(), []
    for ln in (text or "").splitlines():
        base = normalize_line(ln)
        if base and base not in seen:
            seen.add(base)
            out.append(f"- {ln.strip().strip('•-–·* ')}")
    return "\n".join(out)

def summarize_texts(texts: List[str], max_bullets=20) -> str:
    tokens = []
    for t in texts or []:
        for ln in str(t or "").splitlines():
            s = ln.strip(" •-–\t ")
            if s: tokens.append(s)
    uniq = list(dict.fromkeys(tokens))[:max_bullets]
    return "\n".join(f"- {b}" for b in uniq)

# =========================
# GMAIL
# =========================
def _decode_header(s) -> str:
    try: return str(make_header(decode_header(s or "")))
    except: return s or ""

def _strip_html(html: str) -> str:
    t = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", " ", html)
    t = re.sub(r"(?s)<.*?>", " ", t)
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n\s*\n\s*\n+", "\n\n", t)
    return t.strip()

def _message_to_text(msg: email.message.Message) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
                try: return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace")
                except: pass
        for part in msg.walk():
            if part.get_content_type() == "text/html" and "attachment" not in str(part.get("Content-Disposition") or "").lower():
                try: return _strip_html(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
                except: pass
        payloads=[]
        for part in msg.walk():
            if part.get_content_maintype()=="text":
                try: payloads.append(part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", errors="replace"))
                except: pass
        return "\n\n".join(payloads)
    else:
        try:
            payload = msg.get_payload(decode=True)
            if payload is None: return str(msg.get_payload())
            text = payload.decode(msg.get_content_charset() or "utf-8", errors="replace")
            return _strip_html(text) if msg.get_content_type()=="text/html" else text
        except: return str(msg.get_payload())

def fetch_gmail_messages(newer_than_days=30, limit=400) -> List[dict]:
    res=[]
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        return res
    try:
        imap = imaplib.IMAP4_SSL("imap.gmail.com", 993)
        imap.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        ok,_ = imap.select("INBOX")
        if ok!="OK": raise RuntimeError("INBOX non sélectionnable")
        since = (datetime.utcnow() - timedelta(days=newer_than_days)).strftime("%d-%b-%Y")
        ok, data = imap.search(None, f'(SINCE {since})')
        if ok!="OK": return res
        ids = (data[0].split() or [])[-limit:]
        for msg_id in reversed(ids):
            ok, msg_data = imap.fetch(msg_id, "(RFC822)")
            if ok!="OK": continue
            msg  = email.message_from_bytes(msg_data[0][1])
            subj = _decode_header(msg.get("Subject",""))
            body = _message_to_text(msg)
            try:
                dt = parsedate_to_datetime(msg.get("Date"))
                if dt and dt.tzinfo: dt = dt.astimezone().replace(tzinfo=None)
            except:
                dt = datetime.now()
            mid = (msg.get("Message-Id") or msg.get("Message-ID") or "").strip() or f"INBOX:{msg_id.decode()}"
            res.append({"subject": subj, "body": body, "dt": dt, "message_id": mid})
        imap.logout()
    except Exception as e:
        st.error(f"Erreur Gmail IMAP: {e}")
    return res

# =========================
# CLEAN BODY (quotes/signatures)
# =========================
QUOTE_PATTERNS = [
    r"^On .+ wrote:$", r"^Le .+ a écrit :", r"^From: .+$", r"^-{2,}\s*Original Message\s*-{2,}$",
    r"^> .*", r"^__+$", r"^\s*De : .+$", r"^\s*Envoyé : .+$"
]
def strip_quotes_and_signatures(txt: str) -> str:
    lines, out = (txt or "").splitlines(), []
    for ln in lines:
        if any(re.match(p, ln.strip(), re.IGNORECASE) for p in QUOTE_PATTERNS): break
        out.append(ln)
    joined = "\n".join(out)
    joined = re.sub(r"(?is)this message.*?confidential.*", "", joined)
    joined = re.sub(r"(?is)ce message.*?confidentiel.*", "", joined)
    return joined.strip()

# =========================
# PDF LOADER
# =========================
def read_pdf_bytes_to_text(data: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(data))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except:
                pages.append("")
        return "\n".join(pages)
    except Exception as e:
        st.error(f"Lecture PDF: {e}")
        return ""

def guess_date_from_filename(name: str) -> datetime|None:
    m = re.search(r"(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])", name)
    if not m: return None
    try:
        s = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        return datetime.fromisoformat(s)
    except:
        return None

# =========================
# OPENAI (Classifier + Synthèses)
# =========================
from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

DAILY_SYS = (
    "Tu es un assistant financier. Classe TOUT le contenu pertinent dans EXACTEMENT ces 5 catégories.\n"
    "Aucune paraphrase. Recopie le texte source tel quel, en phrases complètes (jamais tronquées).\n"
    "1 item = 1 phrase autonome (pas de virgule finale/parenthèse ouverte).\n"
    "Si des retours à la ligne coupent une phrase, RECOLLE les segments.\n"
    "IMPORTANT : conserve l’ORDRE CHRONOLOGIQUE d’apparition du texte source.\n"
    "IMPORTANT : préfixe chaque item par la date au format [YYYY-MM-DD] si elle est fournie.\n"
    "Exclure signatures/politesse/disclaimers.\n"
    "En français."
)
DAILY_USER_TMPL = """Texte :
---
[[EMAIL_BODY]]
---
Retourne un JSON strict:
{
  "financial_credit_spreads": ["..."],
  "primary_market": ["..."],
  "results": ["..."],
  "ma_and_ratings": ["..."],
  "other_important_infos": ["..."]
}"""

WEEKLY_SYS = (
    "Tu es analyste crédit. À partir d’un contenu agrégé (déjà classé), génère un résumé hebdomadaire PROPRE.\n"
    "Objectifs: dédoublonner, regrouper ce qui se répète, garder chiffres/échéances exacts, zéro hallucination, phrases complètes.\n"
    "Style: fluide, cohérent, structuré. Ne pas tronquer.\n"
)
WEEKLY_USER_TMPL = """Contenu agrégé (par rubrique) :
---
[[WEEKLY_INPUT]]
---
Exemples de sortie attendue 'global_summary' :
{
  "global_summary": " 
  
  Sur le Crédit Financier, les spreads affichent des évolutions contrastées : +1 pdb pour les Seniors bancaires, -1 pdb pour les T2 bancaires, +2 pdb pour les AT1 (-2 pdb pour les €AT1) et, du côté des assureurs, les Seniors étaient à +2 pdb et les Subordonnés à -2 pdb. Globalement, le marché est resté assez stable. Les performances se sont révélées légèrement positives dans l’ensemble : les Seniors, tant bancaires qu’assureurs, ont progressé de +0,1 %, les T2 de +0,2 %, les AT1 de +0,3 % et les Subordonnés assurance de +0,4 %.

Le marché primaire demeure assez calme cette semaine, avec seulement quelques émissions. Du côté des subordonnées, Generali a émis un Tier 2 bullet 11Y (à un faible spread de MS+155pdb), Malakoff Humanis un Tier 2 bullet 10Y (MS+210pdb), BNP a émis €1md de Tier 2 10,5NC5,5 ans à MS+155pdb, Gothaer a émis un Tier 2 20NC10 ans de €300m à MS+250 pdb, et FMO a émis un Tier 2 11NC6 ans de €300m à seulement 75 pdb au-dessus du swap. Côté senior, Julius Baer a émis €500m à 5Y et LHV, basé en Estonie, a émis une petite tranche de €60m€ 4NC3.

La semaine a été animée par de nombreux titres liés aux fusions-acquisitions. Premièrement, après plusieurs jours de « rumeurs » sur le marché, BPCE a officiellement annoncé avoir signé un protocole d’accord pour acquérir NovoBanco, en rachetant 75 % du capital détenu par le fonds de private equity Lone Star Funds. À l’issue de la transaction, le ratio CET1 du Groupe BPCE resterait supérieur à 15 % (contre 16,2 % à fin 2024). La banque est également entrée en négociation avec le gouvernement portugais pour l’acquisition des 25 % restants, à la même valorisation que celle de Lone Star.

Deuxièmement, le PDG d’UniCredit, Andrea Orcel, a indiqué qu’une offre de rachat sur Commerzbank était peu probable aux valorisations actuelles, en raison de la récente hausse du titre. Orcel a suggéré qu’il pourrait attendre jusqu’en 2027 avant d’envisager une acquisition totale. Il prévoit d’obtenir d’ici la fin du mois les autorisations réglementaires pour convertir des dérivés en actions, ce qui pourrait porter la participation d’UniCredit à près de 30 %. Orcel a laissé entendre que cette participation pourrait conférer des « droits et une influence », tout en soulignant que l’incertitude prolongée n’était bénéfique pour aucune des deux parties.

Troisièmement, concernant l’offre en cours sur Banco BPM, Orcel a déclaré qu’il pourrait se retirer si le gouvernement italien ne clarifiait pas les conditions imposées à la transaction. Il estime actuellement à 20 % ou moins la probabilité de poursuivre l’opération.

Enfin, sur le plan des notations, S&P a relevé la note de Sava Re et de Zavarovalnica Sava, deux assureurs slovènes, de A à A+ avec une perspective stable, reflétant leur solide résilience financière et leur position sur le marché. Cette amélioration s’est également traduite par une hausse de la notation des obligations Tier 2 de Sava Re, passant de BBB+ à A-. Cette décision fait suite à l’amélioration de la note souveraine de la Slovénie, passée la semaine dernière de AA- à AA, ce qui a fortement influencé la réévaluation de la position de Sava Re.
  ", or "global_summary": " Sur les dettes subordonnées financières cette semaine, les spreads se sont resserrés sur tous les rangs de la structure de capital : -5bps seniors, -10bps T2, -21bps AT1 (-25bps €AT1), -9bps subordonnées d’assurance. La performance a été positive sur les segments subordonnés : -0.1% seniors bancaires et -0.2% seniors d’assurance, +0.1% T2 et +0.2% subordonnées d’assurance, +0.6% AT1 (+0.7% €AT1).

 Nous avons eu quelques émissions de Subordonnées cette semaine. En Tier 2, Hiscox a émis un 11NC10 pour $500m (coupon 7%, reset 302pdb), en parallèle d'une offre de rachat sur un Tier 2 existant libellé en livres sterling. En AT1, Nationwide a émis un PerpNC6 pour £700m (coupon 7,875%, reset de 359pdb), en parallèle d'une offre de rachat sur deux AT1 existants de Virgin Money. Enfin, Attica, la cinquième plus grande banque grecque, a émis un AT1 (€100m, PerpNC5,5, coupon 9,375%, reset de 727pdb) et un Tier 2 (€150m, 10 ans, coupon 7,375%, reset de 510pdb).

 S&P a relevé la noté de Crelan de A- à A avec une perspective stable.

 Au sujet des fusions & acquisitions, en Pologne, deux des plus grandes institutions financières polonaises (la compagnie d'assurance PZU et la banque Pekao) ont annoncé la signature d'un protocole d'accord concernant la réorganisation du groupe PZU.

 Autrement, le gouvernement suisse a publié ses propositions de réforme bancaire, qui concerneront principalement UBS. La proposition inclut une augmentation du capital détenu localement contre les filiales étrangères, passant de 60 % à 100 %, ce qui, à lui seul, pourrait nécessiter 23 milliards de dollars supplémentaires de capital pour la principale entité basée en Suisse. Les réformes seront mises en place progressivement sur une période de 6 à 8 ans. Le gouvernement a également suggéré une réduction de l’utilisation des AT1 comme capital d’environ 8 milliards de dollars, laissant un besoin supplémentaire de 18 milliards de dollars en capital "alloué à la continuité d’exploitation". Le gouvernement est en train de finaliser le projet de loi, qui sera soumis au Parlement, la nouvelle législation pouvant entrer en vigueur dès 2028 au plus tôt.

 En zone euro, les volumes de prêts ont augmenté de +1,8% sur un an en avril (contre +1,6% sur un an en mars), avec une nouvelle production solide à +11% sur un an, et les nouvelles souscriptions de prêts immobilier en hausse de +40% sur un an (et de manière similaire depuis début 2025). La rémunération des dépôts à terme continue de baisser, en ligne avec les taux de la BCE. Ces indicateurs sont favorables au revenu net d'intérêts des banques, qui sera soutenu par une bonne croissance des volumes et une baisse du coût de financement"

-----
Attendu (JSON strict) :
{
  "global_summary": "Entre 300 et 500 mots. Bien s'inspirer des exemples donnés ci dessous pour la structure. paragraphe structuré, fluide. Un retour à la ligne par thème (m&a, primary market, results, spreads, autres infos).",
  "ma_and_ratings": ["2 à 8 puces propres"],
  "results": ["2 à 8 puces propres"],
  "financial_credit_spreads": ["2 à 8 puces propres"],
  "primary_market": ["2 à 8 puces propres"],
  "other_important_infos": ["2 à 8 puces propres"]
}"""

MONTHLY_SYS = (
    "Tu es analyste crédit. À partir de weeklies, produis une synthèse mensuelle.\n"
    "Objectifs: tendances du mois, thèmes dominants, événements majeurs, risques et pipeline/primaire.\n"
    "Conserve chiffres/échéances exacts, zéro hallucination; phrases complètes.\n"
    "Style: fluide, cohérent, structuré. Ne pas tronquer. Pas de redondance.\n"
)
MONTHLY_USER_TMPL = """Contenu agrégé weeklies (par rubrique) :
---
[[MONTHLY_INPUT]]
---

Exemples de sortie attendue 'global_summary' :
{
  "global_summary": " Mai 2025 :
Le mois de mai a été positif pour les dettes financières, qui poursuivent leur redressement et effacent même complètement le sell-off des droits douaniers en termes de spread.
 
Les spreads bancaires se sont donc resserrés sur le mois : -14 bps sur les Senior, -20 bps sur les Tier 2, et -41 bps sur les AT1 (-30 bps AT1 euros). Pour les assureurs, nous avons eu un resserrement de -11 bps sur les Senior et de -21 bps sur les Subordonnées.
Sur les taux, la courbe € est globalement stable sur le mois, tandis que les courbes £ et $ se sont tendues parallèlement d’environ 20 bps.
Cela s’est traduit par une performance positive sur l’ensemble de la structure de capital, avec une progression de +0,5 % pour les Seniors, +0,7 % pour les Tier 2, +1,6 % pour les AT1 (+1,7 % pour les AT1 en euros), et respectivement +0,4 % et +1,1 % pour les Seniors et Subordonnées d’assurance.
Dans ces conditions, le marché primaire a été particulièrement actif post-publication des résultats, et bien accueilli avec des carnets d’ordres solides.
 
Les résultats du premier trimestre 2025 des institutions financières européennes ont continué d’être excellents, côté banques comme assurances, avec des RoTE entre 10 % et 16 % en moyenne, avec à ce stade des guidances qui ne sont pas affectées par les droits de douane.
 
En matière de fusions-acquisitions, Santander a rejeté l’offre de NatWest pour sa banque de détail britannique, tandis qu'Erste Bank a acquis une participation de 49 % dans Santander Bank Polska. Au Royaume-Uni, Chesnara envisage le rachat de l’activité assurance vie de HSBC UK, et Crédit Agricole a pris une participation de 9,9 % dans la banque belge Crelan. En Islande, Kvika banki a reçu des indications d’intérêt de Arion Banki et Islandbanki, et UniCredit a augmenté sa participation dans Alpha Bank à ~20 %.
 
Les agences de notation ont continué à procéder à des relèvements de notes : Moody’s a relevé la note de Bank of Cyprus à A3, BCP à A2, Novo Banco à Baa1, Monte dei Paschi à Baa1. Fitch a relevé la note de Bank of Ireland à A- et Permanent TSB à BBB, Bank Millennium à BBB-, tout en plaçant sous surveillance évolutive la note de Mediobanca en raison des incertitudes liées au M&A (rachat par Monte ou bien rachat de Banca Generali).
 OR 
Avril 2025 :
Les performances d’avril, stables à légèrement positives, masquent un mois volatil qui a mal commencé, principalement impacté par l’annonce des « tarifs réciproques » le 2 avril. Cela a créé un environnement de marché très négatif jusqu’au 9 avril, date à laquelle une pause de 90 jours a été annoncée. Les marchés se sont ensuite redressés et ont terminé le mois soit stables, soit positifs grâce au resserrement des spreads et à la baisse générale des taux.
 
Les spreads ont récupéré 60% de leur écartement du début du mois d’avril, mais ont tout de même fini plus écartés dans tous les segments de la structure du capital : les Seniors se sont écartés de +10pdb, les Tier 2 IG de +20pdb et les AT1 de +45pdb.
 
En revanche, les taux ont baissé dans toutes les devises G3 : environ -30pdb en €, entre -28 et +11pdb en $ et entre -40 et -23pdb en £. Ainsi, les performances sont positives dans tous les segments sauf les AT1, avec une performance de +1,0%/+1,1% pour les dettes Seniors des banques et assureurs, +0,8% pour les Tier 2 IG et +0,7% pour les dettes Subordonnées des assureurs, tandis que les AT1 ont terminé stables (AT1 en euros) ou légèrement négatifs à -0,3 % (AT1 toutes devises).
 
Le marché primaire a été très calme après la chute des marchés au début du mois, suivie des périodes de blackout liées à la publication des résultats du 1Q25. Quoi qu’il en soit, les rares émissions réalisées ont été bien accueillies et émises avec peu ou pas de prime de nouvelle émission (NIP), ce qui montre la disponibilité de liquidités prêtes à être investies.
 
La BCE a une nouvelle fois abaissé ses taux à 2,25 %, tout en adoptant un ton accommodant, mais en reconnaissant également qu’elle avait désormais atteint la neutralité.
 
En ce qui concerne les résultats, les entreprises européennes du secteur financier et des assurances (FIG) ayant présenté leurs résultats pour le 1Q25 ont affiché de très bonnes performances : BNPP a enregistré un profit net de 2,95 milliards d’euros (-5 % en glissement annuel), Bankinter 270 millions (+35 %), Mapfre 276 millions (+28 %), Nordea 1,2 milliard (-9 %).
 
Les fusions-acquisitions (M&A) restent d'actualité dans le secteur FIG européen, malgré la crise, avec plusieurs annonces de fusion durant le mois. Parmi celles-ci, Mediobanca a lancé une offre publique d’achat pour acquérir Banca Generali, entièrement financée par la vente de sa participation dans Generali. Baloise et Helvetia vont créer le deuxième plus grand assureur suisse avec une part de marché domestique de 20 %. Ageas a acquis esure au Royaume-Uni pour 1,3 milliard de livres sterling, tandis que Banco BPM a réussi à obtenir 90 % du capital d’Anima. Santander envisagerait de vendre sa filiale polonaise. Enfin, BBVA a reçu l’approbation de la CNMC espagnole pour acheter Sabadell.
 
Les agences de notation ont continué à procéder à des relèvements de notes : Fitch et Moody’s ont amélioré la note de mBank, passant de BBB- à BBB et de Baa3 à Ba1 respectivement. Bank Millennium a été relevée à Ba1 contre Ba2, avec une perspective positive. OTP Bank a également été rehaussée à BBB contre BBB- par S&P. La Grèce a été rehaussée à BBB par S&P avec une perspective stable, tandis que les banques grecques ont vu leur note relevée d’un cran par Fitch : NBG et Eurobank à BBB-, et Piraeus et Alpha Bank à BB+. Swedbank a été relevée de Aa3 à Aa2."
  



Attendu (JSON strict) :
{
  "global_summary": "Entre 300 et 600 mots. Inspirés des exemples ci-dessus. Recap des comptes rendus weekly, organisés par les thématiques ma, results, financials_credit_spread, primary and other important infos. FLuide, organisé.).",
  "ma_and_ratings": ["2–10 puces propres"],
  "results": ["2–10 puces propres"],
  "financial_credit_spreads": ["2–10 puces propres"],
  "primary_market": ["2–10 puces propres"],
  "other_important_infos": ["2–10 puces propres"]
}"""

def _flatten_spaces(s: str) -> str:
    return re.sub(r"\s*\n+\s*", " ", str(s or "")).strip()

def _to_items(value) -> list[str]:
    if isinstance(value, list):
        return [_flatten_spaces(x).lstrip("•-–—* ").strip() for x in value if str(x).strip()]
    lines = [ln.rstrip() for ln in str(value or "").splitlines() if ln.strip()]
    items, cur = [], []
    for ln in lines:
        if re.match(r"^\s*[-•–—]\s+", ln):
            if cur: items.append(_flatten_spaces(" ".join(cur)))
            cur = [ln.lstrip().lstrip("-•–—* ").strip()]
        else:
            cur.append(ln.strip())
    if cur: items.append(_flatten_spaces(" ".join(cur)))
    items = [re.sub(r"\s+", " ", it).strip() for it in items if it and it.strip()]
    return items

def classify_with_openai(text: str) -> Dict[str, str]:
    if not client:
        return {k:"" for k in DAILY_FIELDS}
    body = (text or "")[:80000]
    usr  = DAILY_USER_TMPL.replace("[[EMAIL_BODY]]", body)
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type": "json_object"},
            temperature=0.1,
            messages=[{"role":"system","content":DAILY_SYS},{"role":"user","content":usr}],
        )
        parsed = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        st.error(f"OpenAI error (classify): {e}")
        parsed = {}
    cleaned = {}
    for k in DAILY_FIELDS:
        raw = parsed.get(k, "")
        items = _to_items(raw)
        cleaned[k] = "\n".join(items)
    return cleaned

def synth_llm_weekly(weekly_input_text: str) -> Dict[str,str]:
    if not client:
        return {"global_summary": summarize_texts([weekly_input_text], 20), **{k: "" for k in DAILY_FIELDS}}
    usr = WEEKLY_USER_TMPL.replace("[[WEEKLY_INPUT]]", weekly_input_text[:80000])
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type":"json_object"},
            temperature=0.2,
            messages=[{"role":"system","content":WEEKLY_SYS},{"role":"user","content":usr}],
        )
        parsed = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        st.error(f"OpenAI error (weekly): {e}")
        parsed = {}
    out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
    for k in DAILY_FIELDS:
        out[k] = "\n".join(_to_items(parsed.get(k,"")))
    return out

def synth_llm_monthly(monthly_input_text: str) -> Dict[str,str]:
    if not client:
        return {"global_summary": summarize_texts([monthly_input_text], 16), **{k: "" for k in DAILY_FIELDS}}
    usr = MONTHLY_USER_TMPL.replace("[[MONTHLY_INPUT]]", monthly_input_text[:120000])
    try:
        r = client.chat.completions.create(
            model=OPENAI_MODEL,
            response_format={"type":"json_object"},
            temperature=0.2,
            messages=[{"role":"system","content":MONTHLY_SYS},{"role":"user","content":usr}],
        )
        parsed = json.loads(r.choices[0].message.content or "{}")
    except Exception as e:
        st.error(f"OpenAI error (monthly): {e}")
        parsed = {}
    out = {"global_summary": _flatten_spaces(parsed.get("global_summary",""))}
    for k in DAILY_FIELDS:
        out[k] = "\n".join(_to_items(parsed.get(k,"")))
    return out

# =========================
# ONE-SHOT CLASSIFY HELPERS
# =========================
DATE_TAG_RE = re.compile(r"^\s*\[([0-9]{4}-[0-9]{2}-[0-9]{2})\]\s*(.*)")

def build_aggregate_text(docs: List[dict]) -> str:
    chunks = []
    for d in sorted(docs, key=lambda x: x["dt"]):
        ddate = d["dt"].date().isoformat()
        title = (d.get("title") or "").strip()
        text  = (d.get("text")  or "").strip()
        if not text:
            continue
        head = f"[{ddate}] {title}".strip()
        chunks.append(f"{head}\n{text}")
    return "\n\n---\n\n".join(chunks)

def ensure_dash_bullets(s: str) -> str:
    return "\n".join(f"- {ln.strip().lstrip('- ')}" for ln in (s or "").splitlines() if ln.strip())

def sort_items_by_date_prefix(text_block: str) -> str:
    rows = [ln.strip().lstrip("- ").strip() for ln in (text_block or "").splitlines() if ln.strip()]
    tagged = []
    for i, row in enumerate(rows):
        m = DATE_TAG_RE.match(row)
        if m:
            try:
                dt = datetime.fromisoformat(m.group(1)).date()
            except:
                dt = date.min
            txt = m.group(2).strip()
            tagged.append((dt, i, txt))
        else:
            tagged.append((date.min, i, row))
    tagged.sort(key=lambda t: (t[0], t[1]))
    return "\n".join(f"- {t[2]}" for t in tagged)

def classify_all_at_once(docs: List[dict]) -> Dict[str, str]:
    big_text = build_aggregate_text(docs)
    buckets  = classify_with_openai(big_text)  # UN seul appel
    out = {}
    for k in DAILY_FIELDS:
        value = ensure_dash_bullets(buckets.get(k, ""))
        value = dedupe_bullets(value)
        value = sort_items_by_date_prefix(value)
        out[k] = value
    return out

# =========================
# AGRÉGATION SOURCES
# =========================
def search_emails_in_range(start_dt: datetime, end_dt: datetime, subject_filter: str) -> List[dict]:
    results = []
    for m in fetch_gmail_messages(newer_than_days=GMAIL_SEARCH_DAYS, limit=MAX_EMAILS_FETCH):
        if not m.get("dt"): 
            continue
        if subject_filter.lower() not in (m["subject"] or "").lower():
            continue
        if not (start_dt <= m["dt"] <= end_dt):
            continue
        txt = strip_quotes_and_signatures(m["body"])
        results.append({**m, "body": txt})
    results.sort(key=lambda x: x["dt"])
    return results

def collect_sources_for_range(
    start_dt: datetime,
    end_dt: datetime,
    subject_filter: str,
    uploaded_pdfs,
    emails_override: List[dict] | None = None
) -> List[dict]:
    docs = []
    emails = emails_override if emails_override is not None else search_emails_in_range(start_dt, end_dt, subject_filter)
    for m in emails:
        docs.append({"kind":"email","dt":m["dt"],"title":m["subject"],"text":m["body"]})
    for f in (uploaded_pdfs or []):
        content = f.read()
        text = read_pdf_bytes_to_text(content)
        dt_guess = guess_date_from_filename(f.name) or datetime.now()
        docs.append({"kind":"pdf","dt":dt_guess,"title":f.name,"text":text})
    docs.sort(key=lambda x: x["dt"])
    return docs

# =========================
# UI
# =========================
st.sidebar.title("Menu")
page = st.sidebar.radio("Catégories", ["Weekly","Monthly"], index=0)
st.sidebar.caption("💾 JSON dans ./data/")

st.title("🗞️ Hybrids Financial Debts Views — Weekly / Monthly")

# ---- WEEKLY ----
if page=="Weekly":
    st.subheader("Weekly")

    chosen_day = st.date_input("Choisir un jour de la semaine", value=date.today(), format="YYYY-MM-DD")
    monday, sunday = monday_sunday(chosen_day)
    st.caption(f"Semaine: {monday.isoformat()} → {sunday.isoformat()} (thèmes puis chrono du plus ancien au plus récent)")

    # Browse e-mails (semaine)
    start_dt = datetime.combine(monday, datetime.min.time())
    end_dt   = datetime.combine(sunday, datetime.max.time())

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🔍 Parcourir les e-mails (semaine)"):
            emails = search_emails_in_range(start_dt, end_dt, SUBJECT_WEEKLY_FILTER)
            st.session_state["weekly_email_list"] = emails
            st.session_state["weekly_selected_ids"] = {m["message_id"] for m in emails}
    with colB:
        st.caption("Filtre objet: " + SUBJECT_WEEKLY_FILTER)

    if "weekly_email_list" in st.session_state:
        with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
            sel = st.session_state.get("weekly_selected_ids", set())
            new_sel = set(sel)
            for m in st.session_state["weekly_email_list"]:
                mid = m["message_id"]
                checked = st.checkbox(
                    f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
                    key=f"wk_mail_{mid}",
                    value=(mid in sel)
                )
                if checked: new_sel.add(mid)
                else:       new_sel.discard(mid)
                with st.expander("Voir l’aperçu", expanded=False):
                    st.write((m["body"] or "")[:7000] + ("..." if len(m["body"])>7000 else ""))
            st.session_state["weekly_selected_ids"] = new_sel

    # Choix affichage rubriques
    st.markdown("**Rubriques à afficher**")
    cols = st.columns(5); chosen=[]
    for i,f in enumerate(DAILY_FIELDS):
        with cols[i%5]:
            if st.checkbox(f.replace('_',' ').title(), value=True, key=f"w_{f}"): chosen.append(f)
    chosen = chosen or DAILY_FIELDS

    # PDFs optionnels (si pas d'emails, exiger ≥5)
    st.markdown("**Déposez des e-mails/notes au format PDF (optionnel si vous utilisez les e-mails)**")
    pdf_files = st.file_uploader("Glisser-déposer ici", type=["pdf"], accept_multiple_files=True)

    if st.button("⚙️ Générer le Weekly"):
        selected_emails = []
        if "weekly_email_list" in st.session_state:
            ids = st.session_state.get("weekly_selected_ids", set())
            selected_emails = [m for m in st.session_state["weekly_email_list"] if m["message_id"] in ids]

        pdf_cnt = len(pdf_files or [])
        if not selected_emails and pdf_cnt < 5:
            st.error("Sélectionnez au moins **un e-mail** OU déposez **≥ 5 PDFs** pour générer le Weekly.")
        else:
            emails_override = selected_emails if selected_emails else []
            docs = collect_sources_for_range(start_dt, end_dt, SUBJECT_WEEKLY_FILTER, pdf_files, emails_override=emails_override)
            if not docs:
                st.warning("Aucune source sélectionnée/trouvée (vérifiez e-mails sélectionnés ou PDFs).")
            else:
                # --- NOUVEAU : 1 seul appel de tri ---
                categorized = classify_all_at_once(docs)
                weekly_input = "\n\n".join(
                    f"## {k}\n{categorized[k]}" for k in DAILY_FIELDS if categorized.get(k, "").strip()
                )
                # 1 appel de synthèse
                llm = synth_llm_weekly(weekly_input)
                fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
                global_summary = dedupe_bullets(llm.get("global_summary",""))
                week_key = iso_week_key(monday)

                payload = {
                    "week": week_key,
                    "window": [monday.isoformat(), sunday.isoformat()],
                    "fields": fields,
                    "global_summary": global_summary,
                    "sources": [{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
                }
                st.session_state["weekly_preview"] = payload
                st.success("Weekly généré.")

    if "weekly_preview" in st.session_state:
        wk = st.session_state["weekly_preview"]
        st.markdown(f"### Weekly : {wk['week']}")
        st.write("Fenêtre :", " → ".join(wk.get("window", [])) or "N/A")
        with st.expander("Résumé global", expanded=True):
            st.markdown(md_block(wk.get("global_summary","")))
        for k in DAILY_FIELDS:
            if k in chosen:
                with st.expander(k.replace("_"," ").title(), expanded=False):
                    st.markdown(md_block(wk["fields"].get(k,"")))
        with st.expander("Sources utilisées"):
            for s in wk.get("sources", []):
                st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
        if st.button("💾 Enregistrer ce Weekly"):
            save_json(weekly_path(wk['week']), wk); st.success(f"Weekly sauvegardé : {weekly_path(wk['week'])}")

# ---- MONTHLY ----
elif page=="Monthly":
    st.subheader("Monthly")

    today = date.today()
    y = st.number_input("Année", value=today.year, step=1)
    m = st.number_input("Mois", 1, 12, value=today.month, step=1)
    first = date(int(y), int(m), 1)
    nxt   = first + relativedelta(months=1)
    st.caption(f"Période: {first.isoformat()} → {(nxt - timedelta(days=1)).isoformat()} (thèmes puis chrono du plus ancien au plus récent)")

    # Browse e-mails (mois)
    start_dt = datetime.combine(first, datetime.min.time())
    end_dt   = datetime.combine(nxt - timedelta(seconds=1), datetime.max.time())

    colA, colB = st.columns([1,1])
    with colA:
        if st.button("🔍 Parcourir les e-mails (mois)"):
            emails = search_emails_in_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER)
            st.session_state["monthly_email_list"] = emails
            st.session_state["monthly_selected_ids"] = {m["message_id"] for m in emails}
    with colB:
        st.caption("Filtre objet: " + SUBJECT_MONTHLY_FILTER)

    if "monthly_email_list" in st.session_state:
        with st.expander("📬 E-mails trouvés (cliquer pour voir / cocher à inclure)", expanded=True):
            sel = st.session_state.get("monthly_selected_ids", set())
            new_sel = set(sel)
            for m in st.session_state["monthly_email_list"]:
                mid = m["message_id"]
                checked = st.checkbox(
                    f"[{m['dt'].strftime('%Y-%m-%d %H:%M')}] {m['subject']}",
                    key=f"mo_mail_{mid}",
                    value=(mid in sel)
                )
                if checked: new_sel.add(mid)
                else:       new_sel.discard(mid)
                with st.expander("Voir l’aperçu", expanded=False):
                    st.write((m["body"] or "")[:1500] + ("..." if len(m["body"])>1500 else ""))
            st.session_state["monthly_selected_ids"] = new_sel

    # Rubriques à afficher
    st.markdown("**Rubriques à afficher**")
    cols = st.columns(5); chosen=[]
    for i,f in enumerate(DAILY_FIELDS):
        with cols[i%5]:
            if st.checkbox(f.replace('_',' ').title(), value=True, key=f"m_{f}"): chosen.append(f)
    chosen = chosen or DAILY_FIELDS

    # PDFs optionnels
    st.markdown("**(Optionnel) Déposer des PDF**")
    pdf_files = st.file_uploader("Glisser-déposer des PDF (facultatif)", type=["pdf"], accept_multiple_files=True, key="monthly_pdf")

    if st.button("⚙️ Générer le Monthly"):
        emails_override = None
        if "monthly_email_list" in st.session_state:
            ids = st.session_state.get("monthly_selected_ids", set())
            emails_override = [m for m in st.session_state["monthly_email_list"] if m["message_id"] in ids]

        docs = collect_sources_for_range(start_dt, end_dt, SUBJECT_MONTHLY_FILTER, pdf_files, emails_override=emails_override)
        if not docs:
            st.warning("Aucune source sélectionnée/trouvée.")
        else:
            # 1 seul appel de tri
            categorized = classify_all_at_once(docs)
            monthly_input = "\n\n".join(
                f"## {k}\n{categorized[k]}" for k in DAILY_FIELDS if categorized.get(k, "").strip()
            )
            # 1 appel de synthèse mensuelle
            llm = synth_llm_monthly(monthly_input)
            fields = {k: dedupe_bullets(llm.get(k,"")) for k in DAILY_FIELDS}
            global_summary = dedupe_bullets(llm.get("global_summary",""))
            month_key = f"{int(y):04d}-{int(m):02d}"
            payload = {
                "month": month_key,
                "fields": fields,
                "global_summary": global_summary,
                "sources":[{"title":d["title"],"kind":d["kind"],"dt":d["dt"].isoformat()} for d in docs]
            }
            st.session_state["monthly_preview"]=payload
            st.success("Monthly généré.")

    if "monthly_preview" in st.session_state:
        mo = st.session_state["monthly_preview"]
        st.markdown(f"### Monthly : {mo['month']}")
        with st.expander("Résumé global", expanded=True):
            st.markdown(md_block(mo.get("global_summary","")))
        for k in DAILY_FIELDS:
            if k in chosen:
                with st.expander(k.replace("_"," ").title(), expanded=False):
                    st.markdown(md_block(mo["fields"].get(k,"")))
        with st.expander("Sources utilisées"):
            for s in mo.get("sources", []):
                st.write(f"- [{s['kind']}] {s['dt']} — {s['title']}")
        if st.button("💾 Enregistrer ce Monthly"):
            save_json(monthly_path(mo['month']), mo); st.success(f"Monthly sauvegardé : {monthly_path(mo['month'])}")
