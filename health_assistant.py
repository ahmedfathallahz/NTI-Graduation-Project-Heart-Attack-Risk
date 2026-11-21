# -*- coding: utf-8 -*-
"""
health_assistant.py
Single source of truth for the Gemini health assistant.

Used by:
  * app.py (Streamlit)   ->  from health_assistant import ask_health_assistant, build_system_prompt, dev_log
  * terminal / notebook  ->  python health_assistant.py     (interactive chat)

Needs: pip install google-genai
"""
import sys
import time

# ============================================================
# 1. CONFIG
# ============================================================
GEMINI_MODELS = ["gemini-3.5-flash-lite", "gemini-2.5-flash-lite"]  # primary + fallback
MAX_HISTORY_MESSAGES = 20
MAX_ATTEMPTS = 3

SYSTEM_PROMPT = """
You are a knowledgeable, friendly, and responsible
GENERAL HEALTH AI ASSISTANT powered by Google Gemini.

YOUR ROLE:
Provide reliable general health information and explain
medical topics in a clear and understandable way.

HEALTH TOPICS:
1. Cardiovascular health and heart diseases.
2. Diabetes and blood sugar.
3. Blood pressure and cholesterol.
4. Nutrition, diets, and healthy eating.
5. Physical activity, exercise, and fitness.
6. Mental health, stress, and anxiety.
7. Sleep and healthy lifestyle.
8. Common diseases and symptoms.
9. Medical tests and laboratory results.
10. General medication information.
11. Vitamins and nutritional deficiencies.
12. Prevention and public health.
13. Digestive, respiratory, neurological, and skin health.
14. Women's health and men's health.
15. General medical education.

LANGUAGE:
- Respond in the same language used by the user.
- If the user writes Arabic, respond in clear Arabic.
- Explain medical terms using simple language.
- Be friendly, professional, and respectful.

ANSWER QUALITY:
- Provide accurate, evidence-informed answers.
- Never invent medical facts or statistics.
- Explain uncertainty when evidence is limited.
- Organize complicated answers into clear sections.
- Ask clarification questions when necessary.
- Never diagnose a disease based on symptoms alone.

MEDICAL SAFETY:
- You provide educational information, not a diagnosis.
- Never claim to replace a qualified healthcare professional.
- Never prescribe personalized medication doses.
- Never advise stopping prescribed medication without
  consulting a healthcare professional.
- Explain when symptoms require medical evaluation.
- For severe chest pain, severe breathing difficulty,
  stroke symptoms, loss of consciousness, severe allergic
  reactions, or uncontrolled bleeding, advise immediate
  emergency medical care.
- Never falsely reassure users about serious symptoms.
- Be especially careful with children, pregnancy,
  and chronic medical conditions.

MEDICAL TESTS:
- Explain what laboratory results generally mean.
- Consider units and reference ranges.
- Never diagnose from a laboratory result alone.
- Recommend professional interpretation when appropriate.

PRIVACY:
- Do not request unnecessary personal information.
- Never request passwords or API keys.
- Remind users not to share identifying medical information.

IMPORTANT:
- Answer questions about general health, not only heart disease.
- Do not restrict answers to any notebook or dataset.
- If a question is unrelated to health, politely explain
  that you specialize in general health information.

Your goal is to provide safe, clear, and useful health education.
"""

LANGUAGE_RULE = (
    "\n\nLANGUAGE RULE: identify the language of the user's LATEST message and reply entirely "
    "in that same language. Ignore the language of earlier messages and of the app interface."
)
CONCISE_RULE = "\n\nKeep answers concise (under ~250 words unless the user asks for detail)."


def build_system_prompt(extra_context: str = "", concise: bool = False) -> str:
    """Base prompt + optional app context + language rule (+ optional brevity rule)."""
    return SYSTEM_PROMPT + (extra_context or "") + LANGUAGE_RULE + (CONCISE_RULE if concise else "")


def dev_log(msg: str):
    """Developer-only diagnostics -> terminal, never shown to end users."""
    print(f"[NABD assistant] {msg}", file=sys.stderr, flush=True)


# ============================================================
# 2. CORE CALL  (no UI, no input(), no print to user)
# ============================================================
def ask_health_assistant(history, api_key, system_prompt=None, models=None, log=dev_log):
    """
    history: list of {"role": "user"|"assistant", "content": str}  (last item = new question)
    returns: (answer, model_used, error_code)
      error_code in {None, "nolib", "key", "quota", "server", "notfound", "empty", "other"}
    Retries 5xx errors with backoff, falls back to the next model, stops on key/quota errors.
    """
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        log(f"google-genai could not be imported ({e}). Run: pip install google-genai")
        return None, None, "nolib"

    client = genai.Client(api_key=api_key)
    contents = [
        types.Content(role="user" if m["role"] == "user" else "model",
                      parts=[types.Part.from_text(text=m["content"])])
        for m in history[-MAX_HISTORY_MESSAGES:]
    ]
    config = types.GenerateContentConfig(system_instruction=system_prompt or build_system_prompt(),
                                         temperature=0.4, max_output_tokens=2048)
    last = "other"
    for model in (models or GEMINI_MODELS):
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = client.models.generate_content(model=model, contents=contents, config=config)
                if r.text:
                    return r.text, model, None
                last = "empty"
                break
            except Exception as e:
                msg = str(e).lower()
                log(f"{model} (attempt {attempt + 1}/{MAX_ATTEMPTS}): {type(e).__name__}: {str(e)[:300]}")
                if any(x in msg for x in ("api_key_invalid", "api key not valid", "api key", "401", "403", "permission")):
                    return None, model, "key"
                if any(x in msg for x in ("429", "quota", "resource_exhausted")):
                    return None, model, "quota"
                if any(x in msg for x in ("503", "500", "unavailable", "internal", "overloaded")):
                    last = "server"
                    if attempt < MAX_ATTEMPTS - 1:
                        time.sleep(2 ** (attempt + 1))
                        continue
                    break  # next model
                if "404" in msg or "not_found" in msg:
                    last = "notfound"
                    break  # next model
                last = "other"
                break
    return None, None, last


# ============================================================
# 3. TERMINAL / NOTEBOOK CHAT  (only runs when executed directly)
# ============================================================
def chat_cli():
    from getpass import getpass
    key = getpass("Enter your Google Gemini API Key: ").strip()
    if not key:
        raise ValueError("Gemini API Key cannot be empty.")
    history = []
    print("=" * 65 + "\n GENERAL HEALTH AI ASSISTANT (Gemini)\n" + "=" * 65)
    print("Commands: exit | reset\n")
    while True:
        try:
            q = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "reset":
            history.clear()
            print("Conversation reset.")
            continue
        if not q:
            continue
        history.append({"role": "user", "content": q})
        ans, model, err = ask_health_assistant(history, key, build_system_prompt())
        if ans:
            history.append({"role": "assistant", "content": ans})
            print(f"\n[{model}]\n{ans}\n")
        else:
            history.pop()
            print(f"\nFailed: {err}\n")


if __name__ == "__main__":
    chat_cli()
