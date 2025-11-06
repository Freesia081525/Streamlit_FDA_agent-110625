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
# Page Config & New "WOW" Theme
# ==============================================================================
st.set_page_config(page_title="Ferrari FDA Agent", page_icon="🏎️", layout="wide")

# Modern UI CSS with a sleek, dark theme inspired by "WOW" UIs
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
# Defaults and Prompt Library (Same as original)
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
# Session State Init (Same as original)
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
# Utility Functions (Same as original, with minor enhancements)
# ==============================================================================
def read_env_secret(key: str) -> Tuple[bool, Optional[str]]:
    v = os.getenv(key)
    return (True, v) if v else (False, None)

def parse_pdf_to_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as e:
        st.error(f"PDF Parsing Error: {e}")
        return "（PDF 解析失敗或為掃描影像，請改用 OCR 或重新上傳）"

def parse_plain_file(name: str, content: bytes) -> str:
    ext = (name.split(".")[-1] or "").lower()
    try:
        if ext in ["txt", "md", "markdown"]:
            return content.decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Text parsing error for {name}: {e}")
    return ""

def estimate_tokens(text: str) -> int:
    return max(1, int(len(text) / 4))

@st.cache_data
def load_agents_config() -> Dict[str, Any]:
    try:
        config_path = Path("agents.yaml")
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return yaml.safe_load(DEFAULT_AGENTS_YAML)
    except Exception as e:
        st.error(f"Error loading agents config: {e}")
        return yaml.safe_load(DEFAULT_AGENTS_YAML)

# ==============================================================================
# API Clients (Same as original)
# ==============================================================================
def get_model_clients(openai_key: Optional[str], gemini_key: Optional[str], xai_key: Optional[str]) -> Dict[str, Any]:
    clients = {"openai": None, "gemini": None, "grok": None}

    if openai_key:
        try:
            from openai import OpenAI
            clients["openai"] = OpenAI(api_key=openai_key)
            st.session_state.provider_health["openai"] = "OK"
        except Exception as e:
            st.session_state.provider_health["openai"] = f"Init error: {e}"

    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            clients["gemini"] = genai
            st.session_state.provider_health["gemini"] = "OK"
        except Exception as e:
            st.session_state.provider_health["gemini"] = f"Init error: {e}"

    if xai_key:
        try:
            from xai_sdk import Client
            clients["grok"] = Client(api_key=xai_key, timeout=3600)
            st.session_state.provider_health["grok"] = "OK"
        except Exception as e:
            st.session_state.provider_health["grok"] = f"Init error: {e}"

    return clients

def _openai_chat(client, model, prompt, user_input, temperature, max_tokens) -> str:
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role":"system","content":prompt},{"role":"user","content":user_input}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    latency = time.time() - t0
    st.session_state.last_latency["openai"] = latency
    return resp.choices[0].message.content

def _gemini_generate(genai, model, prompt, user_input, temperature, max_tokens) -> str:
    t0 = time.time()
    model_obj = genai.GenerativeModel(model)
    content = f"{prompt}\n\n---\n\n{user_input}"
    resp = model_obj.generate_content(
        content,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
    )
    latency = time.time() - t0
    st.session_state.last_latency["gemini"] = latency
    return resp.text

def _grok_chat(client, model, prompt, user_input, temperature, max_tokens) -> str:
    try:
        t0 = time.time()
        chat = client.chat.create(model=model)
        from xai_sdk.chat import user as xai_user, system as xai_system
        chat.append(xai_system(prompt))
        chat.append(xai_user(user_input))
        response = chat.sample()
        latency = time.time() - t0
        st.session_state.last_latency["grok"] = latency
        return response.content
    except Exception as e:
        return f"[Grok Error] {e}"

def run_agent(agent_def: Dict[str, Any], user_input: str, clients: Dict[str, Any]) -> str:
    model_cfg = agent_def.get("model", {}) or {}
    provider = model_cfg.get("provider", "gemini")
    model = model_cfg.get("name") or "gemini-2.5-flash"
    temperature = float(model_cfg.get("temperature", 0.3))
    max_tokens = int(model_cfg.get("max_tokens", 4096))
    prompt = agent_def.get("prompt", "")

    def auto_provider() -> Optional[str]:
        if clients.get("gemini"): return "gemini"
        if clients.get("openai"): return "openai"
        if clients.get("grok"): return "grok"
        return None

    provider_resolved = provider if provider != "auto" else auto_provider()
    if not provider_resolved:
        return "No model provider configured. Please add at least one API key."

    try:
        if provider_resolved == "openai" and clients.get("openai"):
            return _openai_chat(clients["openai"], model, prompt, user_input, temperature, max_tokens)
        elif provider_resolved == "gemini" and clients.get("gemini"):
            return _gemini_generate(clients["gemini"], model, prompt, user_input, temperature, max_tokens)
        elif provider_resolved == "grok" and clients.get("grok"):
            return _grok_chat(clients["grok"], model, prompt, user_input, temperature, max_tokens)
        else:
            return f"Model provider '{provider_resolved}' is not properly configured."
    except Exception as e:
        return f"Agent execution error: {e}"

def fill_checklist_from_summary(summary_text: str, checklist_json: Dict[str, Any], agent_def: Dict[str, Any], clients: Dict[str, Any]) -> Dict[str, Any]:
    payload = json.dumps({"summary": summary_text, "checklist": checklist_json}, ensure_ascii=False)
    output = run_agent(agent_def, payload, clients)
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)
    if json_match:
        output = json_match.group(1)
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        st.warning("The agent did not return valid JSON. Retaining the original checklist.")
        return checklist_json

def html_status_transform(text: str) -> str:
    result_html = text.replace("[YES]", "<span class='highlight-positive'>✔ YES</span>")
    result_html = result_html.replace("[NO]", "<span class='highlight-negative'>✘ NO</span>")
    result_html = result_html.replace("PASS", "<span class='highlight-positive'>✔ PASS</span>")
    result_html = result_html.replace("FAIL", "<span class='highlight-negative'>✘ FAIL</span>")
    return result_html

def checklist_to_dataframe(checklist: Dict[str, Any]) -> pd.DataFrame:
    records = []
    for section, items in checklist.items():
        for it in items:
            records.append({
                "section": section,
                "審查項目編號": it.get("審查項目編號",""),
                "審查項目名稱": it.get("審查項目名稱",""),
                "填寫內容": it.get("填寫內容","")
            })
    return pd.DataFrame(records)

def get_completion_rate(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    filled = df["填寫內容"].apply(lambda x: str(x).strip() != "" and str(x).strip() not in ["無資料","Not Provided"]).sum()
    return round(100 * filled / len(df), 1)

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

def dict_to_json_bytes(d: dict) -> bytes:
    return json.dumps(d, ensure_ascii=False, indent=2).encode("utf-8")

def df_to_excel_bytes(df: pd.DataFrame) -> bytes:
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Checklist")
    return output.getvalue()


# ==============================================================================
# Sidebar: Revamped with expanders for a cleaner look
# ==============================================================================
with st.sidebar:
    st.title("🏎️ Settings")

    with st.expander("🔐 API Keys", expanded=True):
        gem_env, gem_env_val = read_env_secret("GEMINI_API_KEY")
        gemini_key = st.text_input("Gemini API Key", type="password", value=gem_env_val or "", help="Loaded from environment if available.")
        
        oa_env, oa_env_val = read_env_secret("OPENAI_API_KEY")
        openai_key = st.text_input("OpenAI API Key", type="password", value=oa_env_val or "", help="Loaded from environment if available.")

        xai_env, xai_env_val = read_env_secret("XAI_API_KEY")
        xai_key = st.text_input("XAI API Key (Grok)", type="password", value=xai_env_val or "", help="Loaded from environment if available.")

    with st.expander("🧩 Agents Configuration"):
        agents_file = st.file_uploader("Upload agents.yaml", type=["yaml","yml"])
        if agents_file:
            st.session_state.agents_cfg = yaml.safe_load(agents_file.getvalue())
        else:
            if 'agents_cfg' not in st.session_state or st.session_state.agents_cfg is None:
                st.session_state.agents_cfg = load_agents_config()
        st.info("Upload a custom `agents.yaml` or the app will use its default configuration.")

    with st.expander("⚙️ Provider Health"):
        clients = get_model_clients(gemini_key, openai_key, xai_key)
        ph = st.session_state.provider_health or {}
        for prov, status in ph.items():
            chip_class = "chip-pass" if status == "OK" else "chip-warn"
            st.markdown(f"**{prov.capitalize()}**: <span class='status-chip {chip_class}'>{status}</span>", unsafe_allow_html=True)

# ==============================================================================
# Header
# ==============================================================================
st.title("Ferrari FDA Evidence Extractor + Comparator")
st.markdown("<p style='text-align: center; color: var(--secondary-text-color);'>A YAML-driven multi-agent pipeline with an interactive dashboard, powered by Streamlit.</p>", unsafe_allow_html=True)
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)


# ==============================================================================
# Main Application Flow in Tabs
# ==============================================================================
tab_upload, tab_summary, tab_checklist, tab_agents, tab_dashboard, tab_reports = st.tabs([
    "📂 Upload & Parse", "📝 Summary", "📋 Checklist", "🛠️ Advanced Agents", "📊 Dashboard", "📤 Reports"
])

# 1) Upload & Parse Tab
with tab_upload:
    st.header("Step 1: Provide Your Documentation")
    input_method = st.radio("Choose your input method:", ["Upload Files", "Paste Text"], horizontal=True)

    if input_method == "Upload Files":
        files = st.file_uploader("Upload submission materials (PDF/TXT/MD)", type=["pdf","txt","md","markdown"], accept_multiple_files=True, label_visibility="collapsed")
        if files:
            st.session_state.uploaded_docs = [f.name for f in files]
            parsed_chunks = []
            with st.spinner("Processing files..."):
                for f in files:
                    ext = f.name.split(".")[-1].lower()
                    text = parse_pdf_to_text(f.read()) if ext == "pdf" else parse_plain_file(f.name, f.read())
                    if text:
                        parsed_chunks.append(text)
            st.session_state.parsed_text = "\n\n---\n\n".join(parsed_chunks)
            st.success(f"Successfully loaded and parsed {len(files)} file(s). Total estimated tokens: {estimate_tokens(st.session_state.parsed_text)}")
    else:
        pasted_text = st.text_area("Paste content here", height=300, placeholder="Paste your 510(k) submission text, clinical data, or other relevant content...", label_visibility="collapsed")
        if pasted_text:
            st.session_state.parsed_text = pasted_text
            st.success(f"Text loaded successfully. Estimated tokens: {estimate_tokens(pasted_text)}")
    
    if st.session_state.parsed_text:
        with st.expander("Show Parsed Text Preview"):
            st.text_area("Preview", st.session_state.parsed_text[:6000], height=200, disabled=True)


# 2) Summary & Editor Tab
with tab_summary:
    st.header("Step 2: Generate and Refine the Summary")
    if st.button("🚀 Generate Summary with AI Agent", disabled=not st.session_state.parsed_text.strip()):
        try:
            agents_cfg = st.session_state.agents_cfg or {}
            agent_def = next(a for a in agents_cfg.get("agents", []) if a.get("id") == "summarizer")
            with st.spinner("Generating summary..."):
                output = run_agent(agent_def, st.session_state.parsed_text, clients)
                st.session_state.summary_text = output
            st.toast("Summary generated!", icon="✅")
        except StopIteration:
            st.error("Summarizer agent not found in configuration.")

    st.session_state.summary_text = st.text_area("Editable Summary:", value=st.session_state.summary_text, height=400)


# 3) Checklist Tab
with tab_checklist:
    st.header("Step 3: Auto-fill Regulatory Checklist")
    if st.button("🤖 Auto-fill Checklist from Summary", disabled=not st.session_state.summary_text.strip()):
        try:
            agents_cfg = st.session_state.agents_cfg or {}
            agent_def = next(a for a in agents_cfg.get("agents", []) if a.get("id") == "checklist_filler")
            with st.spinner("Agent is filling the checklist..."):
                st.session_state.checklist_json = fill_checklist_from_summary(
                    st.session_state.summary_text, st.session_state.checklist_json, agent_def, clients
                )
            st.toast("Checklist auto-filled!", icon="📝")
        except StopIteration:
            st.error("Checklist filler agent not found in configuration.")

    df = checklist_to_dataframe(st.session_state.checklist_json)
    st.dataframe(df, use_container_width=True, height=500)
    
    with st.expander("View Raw Checklist JSON"):
        st.json(st.session_state.checklist_json)

# 4) Advanced Agents Tab
with tab_agents:
    st.header("Run Specialized Agents")
    agents_cfg = st.session_state.agents_cfg or {}
    enabled_agents = [a for a in agents_cfg.get("agents", []) if a.get("enabled", True)]
    
    if not enabled_agents:
        st.warning("No enabled agents found in the configuration.")
    else:
        agent_options = {a["name"]: a["id"] for a in enabled_agents}
        selected_agent_name = st.selectbox("Select an agent:", options=list(agent_options.keys()))
        
        if selected_agent_name:
            agent_def = next((a for a in enabled_agents if a["name"] == selected_agent_name), None)
            st.info(f"**Description:** {agent_def.get('description', 'No description provided.')}")
            
            with st.expander("Tune Agent Parameters"):
                # Simplified tuning in columns
                model_cfg = agent_def.get("model", {})
                col1, col2, col3 = st.columns(3)
                edited_provider = col1.selectbox("Provider", ["auto", "gemini", "openai", "grok"], key=f"{agent_def['id']}_prov")
                edited_model = col2.text_input("Model Name", model_cfg.get("name", ""), key=f"{agent_def['id']}_model")
                edited_temp = col3.slider("Temperature", 0.0, 1.0, float(model_cfg.get("temperature", 0.3)), 0.05, key=f"{agent_def['id']}_temp")
            
            if st.button(f"Execute: {selected_agent_name}"):
                default_input = st.session_state.summary_text or st.session_state.parsed_text
                if not default_input.strip():
                    st.error("Input is empty. Please provide data in the 'Upload' tab first.")
                else:
                    with st.spinner(f"Running {selected_agent_name}..."):
                        temp_agent_def = agent_def.copy()
                        temp_agent_def['model'] = {'provider': edited_provider, 'name': edited_model, 'temperature': edited_temp}
                        output = run_agent(temp_agent_def, default_input, clients)
                        st.session_state.latest_agent_output = output
            
            if st.session_state.latest_agent_output:
                st.subheader("Agent Output:")
                try:
                    parsed_output = json.loads(st.session_state.latest_agent_output)
                    st.json(parsed_output)
                except json.JSONDecodeError:
                    st.markdown(html_status_transform(st.session_state.latest_agent_output), unsafe_allow_html=True)

# 5) Dashboard Tab
with tab_dashboard:
    st.header("Project Dashboard")
    
    # KPIs
    total_tokens = estimate_tokens(st.session_state.parsed_text)
    summary_tokens = estimate_tokens(st.session_state.summary_text)
    checklist_df = checklist_to_dataframe(st.session_state.checklist_json)
    completion = get_completion_rate(checklist_df)

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown(f"<div class='kpi-card'><h3>Input Tokens (est.)</h3><p>{total_tokens}</p></div>", unsafe_allow_html=True)
    with k2:
        st.markdown(f"<div class='kpi-card'><h3>Summary Tokens (est.)</h3><p>{summary_tokens}</p></div>", unsafe_allow_html=True)
    with k3:
        st.markdown(f"<div class='kpi-card'><h3>Checklist Completion</h3><p>{completion}%</p></div>", unsafe_allow_html=True)
    
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        st.subheader("Compliance Signals")
        last_output = st.session_state.latest_agent_output or ""
        yes_count = len(re.findall(r"\[YES\]", last_output))
        no_count = len(re.findall(r"\[NO\]", last_output))

        if yes_count > 0 or no_count > 0:
            if PLOTLY_AVAILABLE:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=yes_count,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Positive Compliance Signals"},
                    gauge={'axis': {'range': [None, yes_count + no_count]}, 'bar': {'color': "#4CAF50"}}
                ))
                st.plotly_chart(fig, use_container_width=True)
            else:
                 st.write(f"**YES:** {yes_count} | **NO:** {no_count}")
        else:
            st.info("Run the 'Compliance Checker' agent to see signals here.")

    with colB:
        st.subheader("Agent Latency")
        lat = st.session_state.last_latency
        if lat and PLOTLY_AVAILABLE:
            lat_df = pd.DataFrame(list(lat.items()), columns=['Provider', 'Latency (s)'])
            fig = go.Figure(go.Bar(
                x=lat_df['Latency (s)'],
                y=lat_df['Provider'],
                orientation='h',
                marker_color='#D32F2F'
            ))
            fig.update_layout(title="Last Run Latency", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        elif lat:
            st.json(lat)
        else:
            st.info("Run any agent to see latency metrics here.")

# 6) Reports Tab
with tab_reports:
    st.header("Export Your Results")
    st.info("Download the generated summary, checklist, and full report in various formats.")
    df = checklist_to_dataframe(st.session_state.checklist_json)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.download_button("⬇️ Checklist (CSV)", data=df_to_csv_bytes(df), file_name="checklist.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇️ Checklist (Excel)", data=df_to_excel_bytes(df), file_name="checklist.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    with c3:
        st.download_button("⬇️ Checklist (JSON)", data=dict_to_json_bytes(st.session_state.checklist_json), file_name="checklist.json", mime="application/json", use_container_width=True)
    with c4:
        md_report = f"# FDA Report\n\n## Summary\n\n{st.session_state.summary_text}\n\n## Latest Agent Output\n\n{st.session_state.latest_agent_output}"
        st.download_button("⬇️ Full Report (MD)", data=md_report.encode("utf-8"), file_name="fda_report.md", mime="text/markdown", use_container_width=True)

    with st.expander("Sample `agents.yaml` for reference"):
        st.code(DEFAULT_AGENTS_YAML, language="yaml")
