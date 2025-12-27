import os
import json
import streamlit as st
import google.genai as genai
from google.genai.types import GenerateContentConfig
import yaml
from pathlib import Path
import json

with open("toxicity_bench_sample.json", "r") as f:
    TOXIC_BENCH = json.load(f)

import csv

LOG_PATH = "moderation_log.csv"

def log_run(user_text, gemini_result, openai_result, claude_result, rules_result):
    fieldnames = [
        "text",
        "gemini_decision", "gemini_reason",
        "openai_decision", "openai_reason",
        "claude_decision", "claude_reason",
        "rules_decision", "rules_reason",
    ]

    file_exists = os.path.exists(LOG_PATH)

    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        # Write header only once, when file is first created
        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {
                "text": user_text,
                "gemini_decision": gemini_result["decision"],
                "gemini_reason": gemini_result["reason"],
                "openai_decision": openai_result["decision"],
                "openai_reason": openai_result["reason"],
                "claude_decision": claude_result["decision"],
                "claude_reason": claude_result["reason"],
                "rules_decision": rules_result["decision"],
                "rules_reason": rules_result["reason"],
            }
        )

RULES_PATH = Path(__file__).parent / "rules.yaml"

def load_rules():
    try:
        with open(RULES_PATH, "r") as f:
            data = yaml.safe_load(f)
        return data.get("categories", {})
    except Exception:
        return {}

RULES = load_rules()

def rule_based_moderation(text: str):
    """
    Very simple keyword rules. Returns decision, reason, and raw matches.
    """
    text_lower = text.lower()
    matches = []

    for category, phrases in RULES.items():
        for phrase in phrases:
            if phrase.lower() in text_lower:
                matches.append((category, phrase))

    if matches:
        # Block on first match for demo
        first_cat, first_phrase = matches[0]
        reason = f"Blocked by rules: category='{first_cat}', phrase='{first_phrase}'"
        return {
            "decision": "block",
            "reason": reason,
            "raw": matches,
        }

    return {
        "decision": "allow",
        "reason": "No rule matched.",
        "raw": [],
    }

# --- Gemini setup ---
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.warning(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it in your terminal before running the app."
    )

def moderate_with_gemini(text: str):
    """
    Call a Gemini model to get a simple moderation judgment.
    Returns a dict with: decision, reason, raw.
    """
    if not API_KEY:
        return {
            "decision": "error",
            "reason": "Missing GEMINI_API_KEY",
            "raw": None,
        }

    prompt = f"""
You are a content moderation assistant.

Given the following user text, decide whether it should be ALLOWED or BLOCKED
according to typical online safety rules (hate, harassment, self-harm, sexual content, violence).

Respond ONLY with valid JSON like:
{{
  "decision": "allow" or "block",
  "reason": "short explanation"
}}

User text:
\"\"\"{text}\"\"\"
"""

    try:
        client = genai.Client(api_key=API_KEY)

        response = client.models.generate_content(
            model="gemini-3-pro-preview",  # TODO: make this a dropdown
            contents=prompt,
            config=GenerateContentConfig(),
        )

        raw_text = response.text
        print("RAW GEMINI RESPONSE:", repr(raw_text))

        # Try to parse JSON; if it fails, return a clear error with the raw text
        try:
            data = json.loads(raw_text)
        except Exception as e:
            return {
                "decision": "error",
                "reason": f"Gemini did not return JSON: {e}. Raw text: {raw_text}",
                "raw": raw_text,
            }

        decision = str(data.get("decision", "")).lower()
        reason = str(data.get("reason", ""))

        if decision not in ["allow", "block"]:
            decision = "error"
            reason = f"Unexpected decision value in JSON: {data}"

        return {
            "decision": decision,
            "reason": reason,
            "raw": data,
        }

    except Exception as e:
        print("GEMINI ERROR:", repr(e))
        return {
            "decision": "error",
            "reason": f"Exception while calling Gemini: {e}",
            "raw": None,
        }

# --- Claude setup ---
def moderate_with_claude(text: str):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"decision": "error", "reason": "Missing ANTHROPIC_API_KEY", "raw": None}

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = """
You are a content moderation assistant.

Given USER text, decide whether it should be ALLOWED or BLOCKED according to typical online safety rules (hate, harassment, self-harm, sexual content, violence).

You must respond with valid JSON only, no prose, no markdown, no code fences. The JSON schema is:

{
  "decision": "allow" or "block",
  "reason": "short explanation"
}
"""

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-5-20250929",  # or your chosen ID
            max_tokens=256,
            system=system_prompt,
            messages=[{"role": "user", "content": text}],
        )

        raw_text = msg.content[0].text
        print("RAW CLAUDE RESPONSE:", repr(raw_text))

        raw_text_stripped = raw_text.strip()

        # Try to locate JSON inside any surrounding text
        start = raw_text_stripped.find("{")
        end = raw_text_stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_fragment = raw_text_stripped[start:end+1]
        else:
            json_fragment = raw_text_stripped

        try:
            data = json.loads(json_fragment)
        except json.JSONDecodeError as e:
            return {
                "decision": "error",
                "reason": f"Claude returned non‑JSON text: {raw_text_stripped[:120]}",
                "raw": raw_text,
            }

        decision = str(data.get("decision", "")).lower()
        reason = str(data.get("reason", ""))

        if decision not in ["allow", "block"]:
            return {
                "decision": "error",
                "reason": f"Unexpected decision in Claude JSON: {data}",
                "raw": data,
            }

        return {
            "decision": decision,
            "reason": reason,
            "raw": data,
        }
    except Exception as e:
        return {
            "decision": "error",
            "reason": f"Exception while calling Claude: {e}",
            "raw": None,
        }


# --- OpenAI setup ---
from openai import OpenAI
import anthropic

def moderate_with_openai(text: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"decision": "error", "reason": "Missing OPENAI_API_KEY", "raw": None}

    try:
        client = OpenAI(api_key=api_key)
        resp = client.moderations.create(
            model="omni-moderation-latest",
            input=text,
        )
        result = resp.results[0]

        # `result.categories` is already a dict-like object
        categories = dict(result.categories)
        flagged = any(categories.values())
        flagged_names = [k for k, v in categories.items() if v]

        decision = "block" if flagged else "allow"
        reason = (
            f"Flagged categories: {flagged_names}"
            if flagged_names
            else "No categories flagged."
        )

        return {
            "decision": decision,
            "reason": reason,
            "raw": result.to_dict(),
        }

    except Exception as e:
        return {
            "decision": "error",
            "reason": f"Exception while calling OpenAI: {e}",
            "raw": None,
        }

# --- Streamlit UI ---

st.set_page_config(
    page_title="LLM Moderation Workbench",
    layout="wide",
)

st.title("LLM Moderation Playground")
st.caption("Compare Gemini, OpenAI, Claude, and rules on real and synthetic prompts.")
st.write(
    "Paste some text or pick a benchmark sample, click 'Moderate', "
    "and compare Gemini vs OpenAI vs Claude vs simple rules."
)

st.subheader("Test input")

left, right = st.columns([1, 2])

with left:
    mode = st.radio("Input source", ["Custom", "Benchmark sample"])
    if mode == "Custom":
        st.markdown("Type or paste text on the right.")
    else:
        if not TOXIC_BENCH:
            st.error("toxicity_bench_sample.json is empty or missing.")
        else:
            idx = st.slider("Benchmark index", 0, len(TOXIC_BENCH) - 1, 0)
            sample = TOXIC_BENCH[idx]
            st.markdown(
                f"**Source:** {sample.get('source', 'RealToxicityPrompts')}  \n"
                f"**ID:** {sample.get('id', idx)}  \n"
                f"**Toxicity:** {sample.get('toxicity', 'n/a')}"
            )

with right:
    if mode == "Custom":
        user_text = st.text_area("User content:", height=200)
    else:
        user_text = st.text_area(
            "Benchmark text:",
            value=sample["text"] if TOXIC_BENCH else "",
            height=200,
        )

if st.button("Moderate"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Moderating with all policies..."):
            gemini_result = moderate_with_gemini(user_text)
            openai_result = moderate_with_openai(user_text)
            claude_result = moderate_with_claude(user_text)
            rules_result = rule_based_moderation(user_text)

        # Log this run
        log_run(user_text, gemini_result, openai_result, claude_result, rules_result)

        st.subheader("Decisions")

        rows = [
            {"Policy": "Gemini", "Decision": gemini_result["decision"], "Reason": gemini_result["reason"]},
            {"Policy": "OpenAI", "Decision": openai_result["decision"], "Reason": openai_result["reason"]},
            {"Policy": "Claude", "Decision": claude_result["decision"], "Reason": claude_result["reason"]},
            {"Policy": "Rules", "Decision": rules_result["decision"], "Reason": rules_result["reason"]},
        ]

        st.table(rows)

        with st.expander("Gemini raw data"):
            st.write(gemini_result["raw"])
        with st.expander("OpenAI raw data"):
            st.write(openai_result["raw"])
        with st.expander("Claude raw data"):
            st.write(claude_result["raw"])
        with st.expander("Rules raw matches"):
            st.write(rules_result["raw"])
