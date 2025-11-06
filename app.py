import os
import io
import re
import json
import time
import base64
import yaml
import csv
import streamlit as st
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Optional imports for visualizations
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==============================================================================
# Page Config & Final "WOW" Theme
# ==============================================================================
st.set_page_config(page_title="Ferrari FDA Agent", page_icon="🏎️", layout="wide")

# This is the modern UI CSS with a sleek, dark theme inspired by "WOW" UIs
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

        /* Main theme colors */
        :root {
            --primary-color: #D32F2F; /* Ferrari Red */
            --background-color: #121212;
            --secondary-background-color: #1E1E1E;
            --text-color: #E0E0E0;
            --secondary-text-color: #BDBDBD;
            --card-border-color: #424242;
        }

        html, body, [class*="st-"] {
            font-family: 'Roboto', sans-serif;
        }

        .main {
            background-color: var(--background-color);
            color: var(--text-color);
        }

        h1, h2, h3 {
            color: var(--primary-color);
            text-align: center;
            font-weight: 700;
        }

        .stButton>button {
            border: 2px solid var(--primary-color);
            border-radius: 25px;
            color: var(--primary-color);
            background-color: transparent;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }

        .stButton>button:hover {
            background-color: var(--primary-color);
            color: white;
            transform: scale(1.05);
            box-shadow: 0 0 15px var(--primary-color);
        }

        /* Input widgets */
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--card-border-color);
            border-radius: 8px;
        }

        /* Status highlights */
        .highlight-positive { color: #4CAF50; font-weight: bold; }
        .highlight-negative { color: #FF9800; font-weight: bold; }

        /* Status Chips for a modern look */
        .status-chip {
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            margin: 4px;
            border: 1px solid transparent;
        }
        .chip-pass { background-color: rgba(76, 175, 80, 0.2); color:#4CAF50; border-color: #4CAF50; }
        .chip-fail { background-color: rgba(255, 152, 0, 0.2); color:#FF9800; border-color: #FF9800; }
        .chip-info { background-color: rgba(33, 150, 243, 0.2); color:#2196F3; border-color: #2196F3; }
        .chip-warn { background-color: rgba(244, 67, 54, 0.2); color:#F44336; border-color: #F44336; }

        /* KPI Cards */
        .kpi-card {
            background-color: var(--secondary-background-color);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 1px solid var(--card-border-color);
            transition: transform 0.2s;
        }
        .kpi-card:hover {
            transform: translateY(-5px);
            border-color: var(--primary-color);
        }
        .kpi-card h3 {
            font-size: 1.2em;
            color: var(--secondary-text-color);
            margin: 0;
        }
        .kpi-card p {
            font-size: 2em;
            font-weight: 700;
            color: var(--text-color);
            margin: 5px 0 0 0;
        }

        /* Custom divider */
        .custom-divider {
            margin: 2rem 0;
            height: 1px;
            background: linear-gradient(to right, transparent, var(--primary-color), transparent);
            border: 0;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Defaults and Prompt Library (Original Feature)
# ==============================================================================
PROMPT_LIBRARY = {
    "summarizer": """You are a senior FDA medical device submission summarization expert.
Goals:
- Read the input document text (parsed from PDFs or pasted content).
- Produce a structured, precise, and complete summary in English.
- Cover device description, indications for use, summary of safety and effectiveness, substantial equivalence, clinical data synopsis, labeling notes, and known risks.
- Be concise, but thorough. Cite sections by quoting brief phrases when useful.
Formatting:
- Use clear sections with headings.
- Use bullet points for lists.
- Include a "Key Risks and Mitigations" section at the end.
Quality:
- Avoid hallucinations. If info is missing, explicitly say "Not Provided".
- Do not include internal chain-of-thought; provide final conclusions only.
""",
    "evidence_extractor": """You are a clinical evidence and safety data extraction expert for FDA 510(k) submissions.
Task:
- Extract all clinical evidence, safety data, adverse events, performance testing, bench testing, biocompatibility, sterilization validations, and usability findings.
- Provide source anchors when possible by quoting short text segments.
Output:
- Return a structured JSON object with keys: clinical_evidence, safety_data, adverse_events, performance_testing, bench_testing, biocompatibility, sterilization, usability, notes.
- Use concise bullet-like strings inside lists; no markdown code fences in the JSON text.
""",
    "compliance_checker": """You are an FDA 510(k) compliance checker.
Task:
- Review content and assess compliance across categories: Indications for Use, Substantial Equivalence, Labeling, Device Description, Performance Testing, Biocompatibility, Sterilization, Software/Usability, Risk Management, Clinical Evidence.
- Mark each item as [YES] or [NO] for compliance presence, and provide short justification.
Output:
- Provide a clear, human-readable report. For each category:
  - Category: <name> [YES|NO]
  - Rationale: <1-3 sentences>
- End with a "Summary PASS/FAIL" line with rationale. Avoid chain-of-thought; report findings only.
""",
    "checklist_filler": """You are a regulatory data expert. You will fill in a provided JSON checklist based on the provided 'summary'.
Rules:
- Input is a JSON with "summary" and "checklist".
- Update only the "填寫內容" fields based on the summary.
- If information is not present, fill with "無資料" or "Not Provided".
- Output must be a single valid JSON object only, with the same structure as input 'checklist'. No commentary, no markdown fences.
""",
    "comparator": """You are a device comparator. Build a comparative matrix for the subject device vs predicate/comparator devices.
- Dimensions: Indications, Materials, Dimensions/Specs, Performance, Safety, Sterilization, Software, Clinical Evidence, Labeling Notes.
- If comparator details are missing, note "Not Provided".
- Output a concise markdown table, then a short 3-5 bullet summary of key similarities/differences.
"""
}

DEFAULT_AGENTS_YAML = """
version: 1
agents:
  - id: summarizer
    name: Summary Generator
    description: Generate a complete, structured summary of the submission
    enabled: true
    model: { provider: gemini, name: gemini-2.5-flash, temperature: 0.25, max_tokens: 4096 }
    prompt: |
      """ + PROMPT_LIBRARY["summarizer"] + """
  - id: evidence_extractor
    name: Evidence Extractor
    description: Extract clinical evidence and safety data
    enabled: true
    model: { provider: gemini, name: gemini-2.5-flash-lite, temperature: 0.25, max_tokens: 4096 }
    prompt: |
      """ + PROMPT_LIBRARY["evidence_extractor"] + """
  - id: compliance_checker
    name: Compliance Checker
    description: Check 510(k) compliance across key categories
    enabled: true
    model: { provider: openai, name: gpt-4o-mini, temperature: 0.2, max_tokens: 4096 }
    prompt: |
      """ + PROMPT_LIBRARY["compliance_checker"] + """
  - id: checklist_filler
    name: Checklist Filler
    description: Fill the regulatory checklist JSON based on summary
    enabled: true
    model: { provider: gemini, name: gemini-2.5-flash, temperature: 0.15, max_tokens: 4096 }
    prompt: |
      """ + PROMPT_LIBRARY["checklist_filler"] + """
  - id: comparator
    name: Device Comparator
    description: Create a comparative matrix vs. predicate devices
    enabled: true
    model: { provider: grok, name: grok-4-fast-reasoning, temperature: 0.25, max_tokens: 3000 }
    prompt: |
      """ + PROMPT_LIBRARY["comparator"] + """
"""

DEFAULT_CHECKLIST = {
    "醫療器材查驗登記資料表": [
        {"審查項目編號": "1.1", "審查項目名稱": "主管機關", "填寫內容": ""},
        {"審查項目編號": "1.2", "審查項目名稱": "申請日期", "填寫內容": ""},
        {"審查項目編號": "2.1", "審查項目名稱": "產品名稱", "填寫內容": ""},
        {"審查項目編號": "2.2", "審查項目名稱": "申請公司", "填寫內容": ""},
        {"審查項目編號": "3.1", "審查項目名稱": "適應症", "填寫內容": ""},
        {"審查項目編號": "3.2", "審查項目名稱": "不良事件與風險", "填寫內容": ""}
    ]
}


# ==============================================================================
# Session State Initialization (Original Feature)
# ==============================================================================
if "parsed_text" not in st.session_state: st.session_state.parsed_text = ""
if "summary_text" not in st.session_state: st.session_state.summary_text = ""
if "checklist_json" not in st.session_state: st.session_state.checklist_json = DEFAULT_CHECKLIST
if "latest_agent_output" not in st.session_state: st.session_state.latest_agent_output = ""
if "agents_cfg" not in st.session_state: st.session_state.agents_cfg = None
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = []
if "last_latency" not in st.session_state: st.session_state.last_latency = {}
if "provider_health" not in st.session_state: st.session_state.provider_health = {}

# ==============================================================================
# Utility Functions (With added helper function for robustness)
# ==============================================================================
def read_env_secret(key: str) -> Tuple[bool, Optional[str]]:
    """Checks for an environment variable and returns its status and value."""
    v = os.getenv(key)
    return (True, v) if v else (False, None)

def parse_pdf_to_text(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"PDF Parsing Error: {e}")
        return ""

def parse_plain_file(name: str, content: bytes) -> str:
    """Decodes plain text files."""
    try:
        return content.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Text parsing error for {name}: {e}")
        return ""

def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a string."""
    return max(1, int(len(text) / 4))

@st.cache_data
def load_agents_config() -> Dict[str, Any]:
    """Loads agent configuration from agents.yaml or uses defaults."""
    try:
        config_path = Path("agents.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return yaml.safe_load(DEFAULT_AGENTS_YAML)
    except Exception as e:
        st.error(f"Error loading agents config: {e}")
        return yaml.safe_load(DEFAULT_AGENTS_YAML)

# *** FIX: Robust helper function to prevent AttributeError ***
def get_enabled_agents(agents_cfg: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Safely filters and returns a list of enabled agent dictionaries."""
    if not agents_cfg or "agents" not in agents_cfg:
        return []
    
    potential_agents = agents_cfg.get("agents", [])
    if not isinstance(potential_agents, list):
        st.warning("`agents` key in configuration is not a list. Using empty agent list.")
        return []

    # Filter for items that are dictionaries and are enabled
    return [
        agent for agent in potential_agents
        if isinstance(agent, dict) and agent.get("enabled", True)
    ]

# ==============================================================================
# API Clients and Agent Execution (Original Core Logic)
# ==============================================================================
def get_model_clients(openai_key: Optional[str], gemini_key: Optional[str], xai_key: Optional[str]) -> Dict[str, Any]:
    """Initializes API clients for the available keys."""
    clients = {"openai": None, "gemini": None, "grok": None}
    st.session_state.provider_health = {}

    if openai_key:
        try:
            from openai import OpenAI
            clients["openai"] = OpenAI(api_key=openai_key)
            st.session_state.provider_health["openai"] = "OK"
        except Exception:
            st.session_state.provider_health["openai"] = "Initialization Error"
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            clients["gemini"] = genai
            st.session_state.provider_health["gemini"] = "OK"
        except Exception:
            st.session_state.provider_health["gemini"] = "Initialization Error"
    if xai_key:
        try:
            from xai_sdk import Client
            clients["grok"] = Client(api_key=xai_key, timeout=3600)
            st.session_state.provider_health["grok"] = "OK"
        except Exception:
            st.session_state.provider_health["grok"] = "Initialization Error"
    return clients

def _openai_chat(client, model, prompt, user_input, temperature, max_tokens) -> str:
    t0 = time.time()
    resp = client.chat.completions.create(model=model, messages=[{"role":"system","content":prompt},{"role":"user","content":user_input}], temperature=temperature, max_tokens=max_tokens)
    st.session_state.last_latency["openai"] = time.time() - t0
    return resp.choices[0].message.content

def _gemini_generate(genai, model, prompt, user_input, temperature, max_tokens) -> str:
    t0 = time.time()
    model_obj = genai.GenerativeModel(model)
    content = f"{prompt}\n\n---\n\n{user_input}"
    resp = model_obj.generate_content(content, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
    st.session_state.last_latency["gemini"] = time.time() - t0
    return resp.text

def _grok_chat(client, model, prompt, user_input, temperature, max_tokens) -> str:
    try:
        t0 = time.time()
        from xai_sdk.chat import user as xai_user, system as xai_system
        chat = client.chat.create(model=model)
        chat.append(xai_system(prompt))
        chat.append(xai_user(user_input))
        response = chat.sample()
        st.session_state.last_latency["grok"] = time.time() - t0
        return response.content
    except Exception as e: return f"[Grok Error] {e}"

def run_agent(agent_def: Dict[str, Any], user_input: str, clients: Dict[str, Any]) -> str:
    """Executes a defined agent with its specific model and prompt."""
    model_cfg = agent_def.get("model", {})
    provider = model_cfg.get("provider", "auto")
    model = model_cfg.get("name", "")
    temperature = float(model_cfg.get("temperature", 0.3))
    max_tokens = int(model_cfg.get("max_tokens", 4096))
    prompt = agent_def.get("prompt", "")

    def resolve_provider():
        if provider != "auto": return provider
        if clients.get("gemini"): return "gemini"
        if clients.get("openai"): return "openai"
        if clients.get("grok"): return "grok"
        return None

    provider_resolved = resolve_provider()
    if not provider_resolved or not clients.get(provider_resolved):
        return f"Model provider '{provider_resolved or 'auto'}' is not configured or available. Please check API keys."

    try:
        if provider_resolved == "openai": return _openai_chat(clients["openai"], model, prompt, user_input, temperature, max_tokens)
        if provider_resolved == "gemini": return _gemini_generate(clients["gemini"], model, prompt, user_input, temperature, max_tokens)
        if provider_resolved == "grok": return _grok_chat(clients["grok"], model, prompt, user_input, temperature, max_tokens)
    except Exception as e: return f"Agent execution error with {provider_resolved}: {str(e)}"

# Other utility functions
def fill_checklist_from_summary(summary_text, checklist_json, agent_def, clients):
    payload = json.dumps({"summary": summary_text, "checklist": checklist_json}, ensure_ascii=False)
    output = run_agent(agent_def, payload, clients)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
    if json_match: output = json_match.group(1)
    try: return json.loads(output)
    except json.JSONDecodeError:
        st.warning("Agent did not return valid JSON. Checklist remains unchanged.")
        return checklist_json

def html_status_transform(text: str) -> str: return text.replace("[YES]", "<span class='highlight-positive'>✔ YES</span>").replace("[NO]", "<span class='highlight-negative'>✘ NO</span>").replace("PASS", "<span class='highlight-positive'>✔ PASS</span>").replace("FAIL", "<span class='highlight-negative'>✘ FAIL</span>")
def checklist_to_dataframe(checklist):
    records = [ {"section": sec, **item} for sec, items in checklist.items() for item in items ]
    return pd.DataFrame(records)
def get_completion_rate(df):
    if df.empty: return 0.0
    filled = df["填寫內容"].apply(lambda x: str(x).strip() not in ["", "無資料", "Not Provided"]).sum()
    return round(100 * filled / len(df), 1)
def df_to_csv_bytes(df): return df.to_csv(index=False).encode("utf-8")
def dict_to_json_bytes(d): return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")
def df_to_excel_bytes(df):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer: df.to_excel(writer, index=False, sheet_name="Checklist")
    return output.getvalue()

# ==============================================================================
# Sidebar with Secure API Key Handling and Configuration
# ==============================================================================
with st.sidebar:
    st.title("🏎️ Settings")

    with st.expander("🔐 API Keys", expanded=True):
        st.info("Keys are securely loaded from environment secrets if available.")

        # Gemini
        gem_env, gem_env_val = read_env_secret("GEMINI_API_KEY")
        if gem_env:
            st.success("Gemini API Key loaded from environment.")
            gemini_key = gem_env_val
        else:
            gemini_key = st.text_input("Gemini API Key", type="password", help="Enter key if not set in environment.")

        # OpenAI
        oa_env, oa_env_val = read_env_secret("OPENAI_API_KEY")
        if oa_env:
            st.success("OpenAI API Key loaded from environment.")
            openai_key = oa_env_val
        else:
            openai_key = st.text_input("OpenAI API Key", type="password", help="Enter key if not set in environment.")

        # Grok (xAI)
        xai_env, xai_env_val = read_env_secret("XAI_API_KEY")
        if xai_env:
            st.success("Grok (xAI) API Key loaded from environment.")
            xai_key = xai_env_val
        else:
            xai_key = st.text_input("XAI API Key (Grok)", type="password", help="Enter key if not set in environment.")

    # Initialize clients here to update health status based on keys provided
    clients = get_model_clients(gemini_key, openai_key, xai_key)

    with st.expander("🧩 Agents Configuration"):
        agents_file = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"])
        if agents_file:
            st.session_state.agents_cfg = yaml.safe_load(agents_file.getvalue())
        elif 'agents_cfg' not in st.session_state or st.session_state.agents_cfg is None:
            st.session_state.agents_cfg = load_agents_config()
        st.info("Upload a custom `agents.yaml` or the app will use its built-in defaults.")

    with st.expander("⚙️ Provider Health"):
        ph = st.session_state.provider_health
        for prov in ["gemini", "openai", "grok"]:
            status = ph.get(prov, "Not Configured")
            chip_class = "chip-pass" if status == "OK" else "chip-warn"
            st.markdown(f"**{prov.capitalize()}**: <span class='status-chip {chip_class}'>{status}</span>", unsafe_allow_html=True)

# ==============================================================================
# Main Application Header
# ==============================================================================
st.title("Ferrari FDA Evidence Extractor + Comparator")
st.markdown("<p style='text-align: center; color: var(--secondary-text-color);'>A YAML-driven multi-agent pipeline with an interactive dashboard.</p>", unsafe_allow_html=True)
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# Define tabs for the main interface
tab_upload, tab_summary, tab_checklist, tab_agents, tab_dashboard, tab_reports = st.tabs([
    "📂 Upload & Parse", "📝 Summary", "📋 Checklist", "🛠️ Advanced Agents", "📊 Dashboard", "📤 Reports"
])

# Load and validate the agent configuration for use in the tabs
agents_cfg = st.session_state.agents_cfg
enabled_agents = get_enabled_agents(agents_cfg)

# ==============================================================================
# Tab 1: Upload & Parse
# ==============================================================================
with tab_upload:
    st.header("Step 1: Provide Your Documentation")
    input_method = st.radio("Choose input method:", ["Upload Files", "Paste Text"], horizontal=True)

    if input_method == "Upload Files":
        files = st.file_uploader("Upload materials (PDF/TXT/MD)", type=["pdf", "txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
        if files:
            st.session_state.uploaded_docs = [f.name for f in files]
            parsed_chunks = [parse_pdf_to_text(f.read()) if f.name.endswith('.pdf') else parse_plain_file(f.name, f.read()) for f in files]
            st.session_state.parsed_text = "\n\n---\n\n".join(filter(None, parsed_chunks))
            st.success(f"Successfully processed {len(files)} file(s). Estimated tokens: {estimate_tokens(st.session_state.parsed_text)}")
    else:
        pasted_text = st.text_area("Paste content:", height=300, placeholder="Paste your 510(k) submission text...", label_visibility="collapsed")
        if pasted_text:
            st.session_state.parsed_text = pasted_text
            st.success(f"Text loaded. Estimated tokens: {estimate_tokens(pasted_text)}")

    if st.session_state.parsed_text:
        with st.expander("Show Parsed Text Preview"):
            st.text_area("Preview", st.session_state.parsed_text[:6000], height=200, disabled=True)

# ==============================================================================
# Tab 2: Summary & Editor
# ==============================================================================
with tab_summary:
    st.header("Step 2: Generate and Refine the Summary")
    if st.button("🚀 Generate Summary", disabled=not st.session_state.parsed_text.strip()):
        try:
            agent_def = next(a for a in enabled_agents if a.get("id") == "summarizer")
            with st.spinner("Generating summary..."):
                output = run_agent(agent_def, st.session_state.parsed_text, clients)
                st.session_state.summary_text = output
            st.toast("Summary generated!", icon="✅")
        except StopIteration:
            st.error("A 'summarizer' agent must be enabled in your configuration.")

    st.session_state.summary_text = st.text_area("Editable Summary:", value=st.session_state.summary_text, height=400)

# ==============================================================================
# Tab 3: Checklist
# ==============================================================================
with tab_checklist:
    st.header("Step 3: Auto-fill Regulatory Checklist")
    if st.button("🤖 Auto-fill Checklist", disabled=not st.session_state.summary_text.strip()):
        try:
            agent_def = next(a for a in enabled_agents if a.get("id") == "checklist_filler")
            with st.spinner("Agent is filling the checklist..."):
                st.session_state.checklist_json = fill_checklist_from_summary(st.session_state.summary_text, st.session_state.checklist_json, agent_def, clients)
            st.toast("Checklist auto-filled!", icon="📝")
        except StopIteration:
            st.error("A 'checklist_filler' agent must be enabled in your configuration.")

    df = checklist_to_dataframe(st.session_state.checklist_json)
    st.dataframe(df, use_container_width=True, height=500)
    with st.expander("View Raw Checklist JSON"):
        st.json(st.session_state.checklist_json)

# ==============================================================================
# Tab 4: Advanced Agents
# ==============================================================================
with tab_agents:
    st.header("Run Specialized Agents")
    if not enabled_agents:
        st.warning("No enabled agents found. Check your `agents.yaml` configuration.")
    else:
        agent_options = {a["name"]: a["id"] for a in enabled_agents}
        selected_agent_name = st.selectbox("Select an agent:", options=list(agent_options.keys()))
        agent_def = next((a for a in enabled_agents if a["name"] == selected_agent_name), None)

        if agent_def:
            st.info(f"**Description:** {agent_def.get('description', 'No description provided.')}")
            with st.expander("Tune Agent Parameters"):
                model_cfg = agent_def.get("model", {})
                col1, col2, col3 = st.columns(3)
                edited_provider = col1.selectbox("Provider", ["auto", "gemini", "openai", "grok"], index=0, key=f"{agent_def['id']}_prov")
                edited_model = col2.text_input("Model Name", model_cfg.get("name", ""), key=f"{agent_def['id']}_model")
                edited_temp = col3.slider("Temperature", 0.0, 1.0, float(model_cfg.get("temperature", 0.3)), 0.05, key=f"{agent_def['id']}_temp")

            if st.button(f"Execute: {selected_agent_name}"):
                default_input = st.session_state.summary_text or st.session_state.parsed_text
                if not default_input.strip():
                    st.error("Input is empty. Please provide data in the 'Upload' tab.")
                else:
                    with st.spinner(f"Running {selected_agent_name}..."):
                        temp_agent_def = agent_def.copy()
                        temp_agent_def['model'] = {'provider': edited_provider, 'name': edited_model, 'temperature': edited_temp}
                        output = run_agent(temp_agent_def, default_input, clients)
                        st.session_state.latest_agent_output = output

            if st.session_state.latest_agent_output:
                st.subheader("Last Agent Output:")
                try:
                    st.json(json.loads(st.session_state.latest_agent_output))
                except (json.JSONDecodeError, TypeError):
                    st.markdown(html_status_transform(st.session_state.latest_agent_output), unsafe_allow_html=True)

# ==============================================================================
# Tab 5: Dashboard
# ==============================================================================
with tab_dashboard:
    st.header("Project Dashboard")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"<div class='kpi-card'><h3>Input Tokens (est.)</h3><p>{estimate_tokens(st.session_state.parsed_text)}</p></div>", unsafe_allow_html=True)
    k2.markdown(f"<div class='kpi-card'><h3>Summary Tokens (est.)</h3><p>{estimate_tokens(st.session_state.summary_text)}</p></div>", unsafe_allow_html=True)
    k3.markdown(f"<div class='kpi-card'><h3>Checklist Completion</h3><p>{get_completion_rate(checklist_to_dataframe(st.session_state.checklist_json))}%</p></div>", unsafe_allow_html=True)
    
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Compliance Signals")
        last_output = st.session_state.latest_agent_output or ""
        yes_count = len(re.findall(r"\[YES\]", last_output))
        no_count = len(re.findall(r"\[NO\]", last_output))

        if (yes_count > 0 or no_count > 0) and PLOTLY_AVAILABLE:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=yes_count, title={'text': "Positive Signals"}, gauge={'axis': {'range': [None, yes_count + no_count]}, 'bar': {'color': "#4CAF50"}}))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the 'Compliance Checker' agent to see signals.")
    with colB:
        st.subheader("Agent Latency")
        lat = st.session_state.last_latency
        if lat and PLOTLY_AVAILABLE:
            lat_df = pd.DataFrame(list(lat.items()), columns=['Provider', 'Latency (s)'])
            fig = go.Figure(go.Bar(x=lat_df['Latency (s)'], y=lat_df['Provider'], orientation='h', marker_color='#D32F2F'))
            fig.update_layout(title="Last Run Latency", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        elif lat: st.json(lat)
        else: st.info("Run any agent to see latency metrics.")

# ==============================================================================
# Tab 6: Reports & Downloads
# ==============================================================================
with tab_reports:
    st.header("Export Your Results")
    st.info("Download the generated summary, checklist, and full report.")
    df = checklist_to_dataframe(st.session_state.checklist_json)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("⬇️ Checklist (CSV)", data=df_to_csv_bytes(df), file_name="checklist.csv", mime="text/csv", use_container_width=True)
    c2.download_button("⬇️ Checklist (Excel)", data=df_to_excel_bytes(df), file_name="checklist.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    c3.download_button("⬇️ Checklist (JSON)", data=dict_to_json_bytes(st.session_state.checklist_json), file_name="checklist.json", mime="application/json", use_container_width=True)
    md_report = f"# FDA Report\n\n## Summary\n\n{st.session_state.summary_text}\n\n## Latest Agent Output\n\n{st.session_state.latest_agent_output}"
    c4.download_button("⬇️ Full Report (MD)", data=md_report.encode("utf-8"), file_name="fda_report.md", mime="text/markdown", use_container_width=True)

    with st.expander("View Sample `agents.yaml`"):
        st.code(DEFAULT_AGENTS_YAML, language="yaml")
