def synth_llm_weekly(weekly_input_text: str) -> Dict[str,str]:
    if not client:
        return {"global_summary": summarize_texts([weekly_input_text], 12), **{k: "" for k in DAILY_FIELDS}}
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
