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
from collections import Counter
import networkx as nx

# Optional imports for visualizations
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

# ==============================================================================
# Theme Definitions - 20 Animal Themes + Base Light/Dark
# ==============================================================================
ANIMAL_THEMES = {
    "Ferrari": {"primary": "#D32F2F", "secondary": "#1E1E1E", "accent": "#FF5252", "emoji": "🏎️"},
    "Lion": {"primary": "#FFA726", "secondary": "#5D4037", "accent": "#FFD54F", "emoji": "🦁"},
    "Eagle": {"primary": "#8D6E63", "secondary": "#3E2723", "accent": "#BCAAA4", "emoji": "🦅"},
    "Dolphin": {"primary": "#29B6F6", "secondary": "#01579B", "accent": "#81D4FA", "emoji": "🐬"},
    "Tiger": {"primary": "#FF6F00", "secondary": "#E65100", "accent": "#FFB74D", "emoji": "🐯"},
    "Panda": {"primary": "#424242", "secondary": "#212121", "accent": "#9E9E9E", "emoji": "🐼"},
    "Phoenix": {"primary": "#D32F2F", "secondary": "#B71C1C", "accent": "#EF5350", "emoji": "🔥"},
    "Wolf": {"primary": "#607D8B", "secondary": "#37474F", "accent": "#90A4AE", "emoji": "🐺"},
    "Butterfly": {"primary": "#AB47BC", "secondary": "#6A1B9A", "accent": "#CE93D8", "emoji": "🦋"},
    "Owl": {"primary": "#795548", "secondary": "#4E342E", "accent": "#A1887F", "emoji": "🦉"},
    "Shark": {"primary": "#0277BD", "secondary": "#01579B", "accent": "#4FC3F7", "emoji": "🦈"},
    "Fox": {"primary": "#EF6C00", "secondary": "#E65100", "accent": "#FF9800", "emoji": "🦊"},
    "Peacock": {"primary": "#00897B", "secondary": "#00695C", "accent": "#4DB6AC", "emoji": "🦚"},
    "Elephant": {"primary": "#78909C", "secondary": "#546E7A", "accent": "#B0BEC5", "emoji": "🐘"},
    "Dragon": {"primary": "#C62828", "secondary": "#B71C1C", "accent": "#EF5350", "emoji": "🐉"},
    "Penguin": {"primary": "#263238", "secondary": "#000000", "accent": "#546E7A", "emoji": "🐧"},
    "Flamingo": {"primary": "#EC407A", "secondary": "#C2185B", "accent": "#F48FB1", "emoji": "🦩"},
    "Cheetah": {"primary": "#F9A825", "secondary": "#F57F17", "accent": "#FDD835", "emoji": "🐆"},
    "Octopus": {"primary": "#5E35B1", "secondary": "#4527A0", "accent": "#9575CD", "emoji": "🐙"},
    "Koala": {"primary": "#90A4AE", "secondary": "#607D8B", "accent": "#CFD8DC", "emoji": "🐨"}
}

# Language translations
TRANSLATIONS = {
    "en": {
        "title": "Ferrari FDA Evidence Extractor + Comparator",
        "subtitle": "A YAML-driven multi-agent pipeline with an interactive dashboard",
        "settings": "Settings",
        "theme": "Theme",
        "language": "Language",
        "api_keys": "API Keys",
        "upload_parse": "Upload & Parse",
        "summary": "Summary",
        "checklist": "Checklist",
        "agents": "Advanced Agents",
        "dashboard": "Dashboard",
        "reports": "Reports",
        "agent_config": "Agent Configuration",
        "select_agents": "Select Agents to Use",
        "agent_pipeline": "Agent Pipeline",
        "execute": "Execute",
        "input": "Input",
        "output": "Output",
        "modify": "Modify",
        "next": "Next",
        "completion_rate": "Completion Rate",
        "tokens": "Tokens",
        "latency": "Latency",
        "generate": "Generate",
        "download": "Download",
        "light": "Light Mode",
        "dark": "Dark Mode"
    },
    "zh": {
        "title": "Ferrari FDA 證據提取器 + 比較器",
        "subtitle": "基於 YAML 驅動的多代理管道與互動式儀表板",
        "settings": "設定",
        "theme": "主題",
        "language": "語言",
        "api_keys": "API 金鑰",
        "upload_parse": "上傳與解析",
        "summary": "摘要",
        "checklist": "檢查清單",
        "agents": "進階代理",
        "dashboard": "儀表板",
        "reports": "報告",
        "agent_config": "代理配置",
        "select_agents": "選擇要使用的代理",
        "agent_pipeline": "代理管道",
        "execute": "執行",
        "input": "輸入",
        "output": "輸出",
        "modify": "修改",
        "next": "下一步",
        "completion_rate": "完成率",
        "tokens": "標記",
        "latency": "延遲",
        "generate": "生成",
        "download": "下載",
        "light": "淺色模式",
        "dark": "深色模式"
    }
}

# ==============================================================================
# Page Config
# ==============================================================================
st.set_page_config(page_title="Ferrari FDA Agent", page_icon="🏎️", layout="wide")

# ==============================================================================
# Default Sample Agents Configuration
# ==============================================================================
DEFAULT_SAMPLE_AGENTS = """version: 1
agents:
  - id: summarizer
    name: Summary Generator
    description: Generate a complete, structured summary of the submission
    enabled: true
    model:
      provider: gemini
      name: gemini-2.5-flash
      temperature: 0.25
      max_tokens: 4096
    prompt: |
      You are a senior FDA medical device submission summarization expert.
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
  - id: evidence_extractor
    name: Evidence Extractor
    description: Extract clinical evidence and safety data
    enabled: true
    model:
      provider: openai
      name: gpt-4o-mini
      temperature: 0.25
      max_tokens: 4096
    prompt: |
      You are a clinical evidence and safety data extraction expert for FDA 510(k) submissions.
      Task:
      - Extract all clinical evidence, safety data, adverse events, performance testing, bench testing, biocompatibility, sterilization validations, and usability findings.
      - Provide source anchors when possible by quoting short text segments.
      Output:
      - Return a structured JSON object with keys: clinical_evidence, safety_data, adverse_events, performance_testing, bench_testing, biocompatibility, sterilization, usability, notes.
      - Use concise bullet-like strings inside lists; no markdown code fences in the JSON text.
  - id: compliance_checker
    name: Compliance Checker
    description: Check 510(k) compliance across key categories
    enabled: true
    model:
      provider: gemini
      name: gemini-2.5-flash
      temperature: 0.2
      max_tokens: 4096
    prompt: |
      You are an FDA 510(k) compliance checker.
      Task:
      - Review content and assess compliance across categories: Indications for Use, Substantial Equivalence, Labeling, Device Description, Performance Testing, Biocompatibility, Sterilization, Software/Usability, Risk Management, Clinical Evidence.
      - Mark each item as [YES] or [NO] for compliance presence, and provide short justification.
      Output:
      - Provide a clear, human-readable report. For each category:
        - Category: <name> [YES|NO]
        - Rationale: <1-3 sentences>
      - End with a "Summary PASS/FAIL" line with rationale. Avoid chain-of-thought; report findings only.
"""
if "theme_mode" not in st.session_state: st.session_state.theme_mode = "dark"
if "theme_style" not in st.session_state: st.session_state.theme_style = "Ferrari"
if "language" not in st.session_state: st.session_state.language = "en"
if "parsed_text" not in st.session_state: st.session_state.parsed_text = ""
if "summary_text" not in st.session_state: st.session_state.summary_text = ""
if "checklist_json" not in st.session_state: st.session_state.checklist_json = {}
if "latest_agent_output" not in st.session_state: st.session_state.latest_agent_output = ""
if "agents_cfg" not in st.session_state: st.session_state.agents_cfg = None
if "selected_agents" not in st.session_state: st.session_state.selected_agents = []
if "agent_outputs" not in st.session_state: st.session_state.agent_outputs = {}
if "pipeline_results" not in st.session_state: st.session_state.pipeline_results = []
if "uploaded_docs" not in st.session_state: st.session_state.uploaded_docs = []
if "last_latency" not in st.session_state: st.session_state.last_latency = {}
if "provider_health" not in st.session_state: st.session_state.provider_health = {}

# ==============================================================================
# Dynamic Theme CSS Generator
# ==============================================================================
def get_theme_css(mode: str, style: str) -> str:
    """Generate dynamic CSS based on theme mode and style"""
    theme = ANIMAL_THEMES.get(style, ANIMAL_THEMES["Ferrari"])
    
    if mode == "light":
        bg_color = "#FFFFFF"
        secondary_bg = "#F5F5F5"
        text_color = "#212121"
        secondary_text = "#757575"
        card_border = "#E0E0E0"
    else:
        bg_color = "#121212"
        secondary_bg = "#1E1E1E"
        text_color = "#E0E0E0"
        secondary_text = "#BDBDBD"
        card_border = "#424242"
    
    return f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');
        :root {{
            --primary-color: {theme["primary"]};
            --accent-color: {theme["accent"]};
            --background-color: {bg_color};
            --secondary-background-color: {secondary_bg};
            --text-color: {text_color};
            --secondary-text-color: {secondary_text};
            --card-border-color: {card_border};
        }}
        html, body, [class*="st-"] {{
            font-family: 'Roboto', sans-serif;
        }}
        .main {{
            background-color: var(--background-color);
            color: var(--text-color);
        }}
        h1, h2, h3 {{
            color: var(--primary-color);
            text-align: center;
            font-weight: 700;
        }}
        .stButton>button {{
            border: 2px solid var(--primary-color);
            border-radius: 25px;
            color: var(--primary-color);
            background-color: transparent;
            padding: 10px 24px;
            font-weight: bold;
            transition: all 0.3s ease-in-out;
        }}
        .stButton>button:hover {{
            background-color: var(--primary-color);
            color: white;
            transform: scale(1.05);
            box-shadow: 0 0 15px var(--primary-color);
        }}
        .stTextInput>div>div>input, .stTextArea>div>div>textarea, .stSelectbox>div>div>div {{
            background-color: var(--secondary-background-color);
            color: var(--text-color);
            border: 1px solid var(--card-border-color);
            border-radius: 8px;
        }}
        .highlight-positive {{ color: #4CAF50; font-weight: bold; }}
        .highlight-negative {{ color: #FF9800; font-weight: bold; }}
        .status-chip {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
            margin: 4px;
            border: 1px solid transparent;
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.7; }}
        }}
        
        .chip-pass {{ background-color: rgba(76, 175, 80, 0.2); color:#4CAF50; border-color: #4CAF50; }}
        .chip-fail {{ background-color: rgba(255, 152, 0, 0.2); color:#FF9800; border-color: #FF9800; }}
        .chip-info {{ background-color: rgba(33, 150, 243, 0.2); color:#2196F3; border-color: #2196F3; }}
        .chip-warn {{ background-color: rgba(244, 67, 54, 0.2); color:#F44336; border-color: #F44336; }}
        .chip-active {{ background-color: rgba(156, 39, 176, 0.2); color:#9C27B0; border-color: #9C27B0; }}
        .kpi-card {{
            background: linear-gradient(135deg, var(--secondary-background-color) 0%, var(--background-color) 100%);
            border-radius: 15px;
            padding: 20px;
            text-align: center;
            border: 2px solid var(--card-border-color);
            transition: all 0.3s;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .kpi-card:hover {{
            transform: translateY(-5px) scale(1.02);
            border-color: var(--primary-color);
            box-shadow: 0 8px 12px var(--accent-color);
        }}
        
        .kpi-card h3 {{
            font-size: 1.2em;
            color: var(--secondary-text-color);
            margin: 0;
        }}
        
        .kpi-card p {{
            font-size: 2em;
            font-weight: 700;
            color: var(--primary-color);
            margin: 5px 0 0 0;
        }}
        .custom-divider {{
            margin: 2rem 0;
            height: 2px;
            background: linear-gradient(to right, transparent, var(--primary-color), transparent);
            border: 0;
        }}
        
        .agent-card {{
            background-color: var(--secondary-background-color);
            border-radius: 12px;
            padding: 15px;
            border-left: 4px solid var(--primary-color);
            margin: 10px 0;
            transition: all 0.3s;
        }}
        
        .agent-card:hover {{
            box-shadow: 0 4px 12px var(--accent-color);
            transform: translateX(5px);
        }}
        
        .pipeline-step {{
            background-color: var(--secondary-background-color);
            border-radius: 10px;
            padding: 20px;
            margin: 15px 0;
            border: 2px solid var(--card-border-color);
            position: relative;
        }}
        
        .pipeline-step::before {{
            content: '';
            position: absolute;
            left: -15px;
            top: 50%;
            width: 10px;
            height: 10px;
            background-color: var(--primary-color);
            border-radius: 50%;
            animation: blink 1.5s infinite;
        }}
        
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
        }}
        
        .word-cloud {{
            text-align: center;
            padding: 20px;
        }}
        
        .word-item {{
            display: inline-block;
            margin: 5px;
            padding: 8px 12px;
            background-color: var(--accent-color);
            color: white;
            border-radius: 15px;
            font-weight: bold;
            transition: all 0.3s;
        }}
        
        .word-item:hover {{
            transform: scale(1.2);
            box-shadow: 0 4px 8px var(--primary-color);
        }}
    </style>
    """

# Apply theme
st.markdown(get_theme_css(st.session_state.theme_mode, st.session_state.theme_style), unsafe_allow_html=True)

# ==============================================================================
# Translation Helper
# ==============================================================================
def t(key: str) -> str:
    """Get translation for current language"""
    return TRANSLATIONS[st.session_state.language].get(key, key)

# ==============================================================================
# Utility Functions
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
                config = yaml.safe_load(f)
                # Validate structure
                if config and isinstance(config, dict) and "agents" in config:
                    if isinstance(config["agents"], list):
                        return config
                    else:
                        st.warning("Invalid agents.yaml structure: 'agents' is not a list")
                        return {"version": 1, "agents": []}
                else:
                    st.warning("Invalid agents.yaml structure: missing 'agents' key")
                    return {"version": 1, "agents": []}
        # Return default empty config if file doesn't exist
        return {"version": 1, "agents": []}
    except Exception as e:
        st.error(f"Error loading agents config: {e}")
        return {"version": 1, "agents": []}

def get_enabled_agents(agents_cfg: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Safely filters and returns a list of enabled agent dictionaries."""
    if not agents_cfg or "agents" not in agents_cfg:
        return []
    
    potential_agents = agents_cfg.get("agents", [])
    if not isinstance(potential_agents, list):
        return []

    return [
        agent for agent in potential_agents
        if isinstance(agent, dict) and agent.get("enabled", True)
    ]

# ==============================================================================
# Word Cloud & Network Graph Functions
# ==============================================================================
def extract_keywords(text: str, top_n: int = 30) -> List[Tuple[str, int]]:
    """Extract top keywords from text"""
    # Simple keyword extraction (can be enhanced with NLP libraries)
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    # Filter out common words
    stop_words = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 'will', 'your', 'their', 'which', 'about', 'would', 'there', 'could', 'should'}
    filtered_words = [w for w in words if w not in stop_words]
    counter = Counter(filtered_words)
    return counter.most_common(top_n)

def create_word_cloud_viz(text: str) -> str:
    """Create HTML word cloud visualization"""
    keywords = extract_keywords(text)
    if not keywords:
        return "<p>No keywords to display</p>"
    
    max_count = keywords[0][1]
    html = "<div class='word-cloud'>"
    
    for word, count in keywords:
        size = 0.8 + (count / max_count) * 1.5
        html += f"<span class='word-item' style='font-size: {size}em;'>{word} ({count})</span>"
    
    html += "</div>"
    return html

def create_agent_network_graph(pipeline_results: List[Dict]) -> Any:
    """Create network graph of agent pipeline"""
    if not PLOTLY_AVAILABLE or not pipeline_results:
        return None
    
    G = nx.DiGraph()
    
    for i, result in enumerate(pipeline_results):
        agent_name = result.get("agent_name", f"Agent {i+1}")
        G.add_node(agent_name, step=i)
        if i > 0:
            prev_agent = pipeline_results[i-1].get("agent_name", f"Agent {i}")
            G.add_edge(prev_agent, agent_name)
    
    pos = nx.spring_layout(G)
    
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=2, color='#888'),
        hoverinfo='none',
        mode='lines')
    
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="top center",
        hoverinfo='text',
        marker=dict(
            size=30,
            color='#D32F2F',
            line=dict(width=2, color='#FFFFFF')))
    
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=0,l=0,r=0,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       paper_bgcolor='rgba(0,0,0,0)',
                       plot_bgcolor='rgba(0,0,0,0)'))
    
    return fig

# ==============================================================================
# API Clients and Agent Execution
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
            st.session_state.provider_health["openai"] = "Error"
    if gemini_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            clients["gemini"] = genai
            st.session_state.provider_health["gemini"] = "OK"
        except Exception:
            st.session_state.provider_health["gemini"] = "Error"
    if xai_key:
        try:
            from xai_sdk import Client
            clients["grok"] = Client(api_key=xai_key, timeout=3600)
            st.session_state.provider_health["grok"] = "OK"
        except Exception:
            st.session_state.provider_health["grok"] = "Error"
    return clients

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
        return f"Provider '{provider_resolved or 'auto'}' not configured."

    try:
        t0 = time.time()
        if provider_resolved == "openai":
            resp = clients["openai"].chat.completions.create(
                model=model,
                messages=[{"role":"system","content":prompt},{"role":"user","content":user_input}],
                temperature=temperature,
                max_tokens=max_tokens)
            output = resp.choices[0].message.content
        elif provider_resolved == "gemini":
            model_obj = clients["gemini"].GenerativeModel(model)
            content = f"{prompt}\n\n---\n\n{user_input}"
            resp = model_obj.generate_content(content, 
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
            output = resp.text
        elif provider_resolved == "grok":
            from xai_sdk.chat import user as xai_user, system as xai_system
            chat = clients["grok"].chat.create(model=model)
            chat.append(xai_system(prompt))
            chat.append(xai_user(user_input))
            response = chat.sample()
            output = response.content
        
        st.session_state.last_latency[provider_resolved] = time.time() - t0
        return output
    except Exception as e:
        return f"Execution error: {str(e)}"

# ==============================================================================
# Initialize API Keys (before sidebar)
# ==============================================================================
# Initialize API keys from environment or set to None
gem_env, gem_env_val = read_env_secret("GEMINI_API_KEY")
gemini_key = gem_env_val if gem_env else None

oa_env, oa_env_val = read_env_secret("OPENAI_API_KEY")
openai_key = oa_env_val if oa_env else None

xai_env, xai_env_val = read_env_secret("XAI_API_KEY")
xai_key = xai_env_val if xai_env else None

# ==============================================================================
# Sidebar
# ==============================================================================
with st.sidebar:
    emoji = ANIMAL_THEMES[st.session_state.theme_style]["emoji"]
    st.title(f"{emoji} {t('settings')}")

    # Theme Settings
    with st.expander("🎨 " + t('theme'), expanded=True):
        mode_col, style_col = st.columns(2)
        with mode_col:
            new_mode = st.selectbox(
                t('theme'),
                ["light", "dark"],
                index=0 if st.session_state.theme_mode == "light" else 1,
                key="mode_select"
            )
            if new_mode != st.session_state.theme_mode:
                st.session_state.theme_mode = new_mode
                st.rerun()
        
        with style_col:
            new_style = st.selectbox(
                "Style",
                list(ANIMAL_THEMES.keys()),
                index=list(ANIMAL_THEMES.keys()).index(st.session_state.theme_style),
                format_func=lambda x: f"{ANIMAL_THEMES[x]['emoji']} {x}",
                key="style_select"
            )
            if new_style != st.session_state.theme_style:
                st.session_state.theme_style = new_style
                st.rerun()
    
    # Language Settings
    with st.expander("🌐 " + t('language')):
        new_lang = st.radio(
            t('language'),
            ["en", "zh"],
            index=0 if st.session_state.language == "en" else 1,
            format_func=lambda x: "English" if x == "en" else "繁體中文",
            horizontal=True
        )
        if new_lang != st.session_state.language:
            st.session_state.language = new_lang
            st.rerun()

    # API Keys
    with st.expander("🔐 " + t('api_keys'), expanded=not (gem_env or oa_env or xai_env)):
        if gem_env:
            st.success("✅ Gemini API Key loaded from environment")
        else:
            gemini_key_input = st.text_input("Gemini API Key", type="password", key="gemini_input")
            if gemini_key_input:
                gemini_key = gemini_key_input

        if oa_env:
            st.success("✅ OpenAI API Key loaded from environment")
        else:
            openai_key_input = st.text_input("OpenAI API Key", type="password", key="openai_input")
            if openai_key_input:
                openai_key = openai_key_input

        if xai_env:
            st.success("✅ Grok API Key loaded from environment")
        else:
            xai_key_input = st.text_input("XAI API Key (Grok)", type="password", key="xai_input")
            if xai_key_input:
                xai_key = xai_key_input

# Initialize clients with API keys
    clients = get_model_clients(openai_key, gemini_key, xai_key)
    # Provider Health
    with st.expander("⚙️ Provider Health"):
         ph = st.session_state.provider_health
         for prov in ["gemini", "openai", "grok"]:
             status = ph.get(prov, "Not Configured")
             chip_class = "chip-pass" if status == "OK" else "chip-warn"
             st.markdown(f"**{prov}**: <span class='status-chip {chip_class}'>{status}</span>", unsafe_allow_html=True)
# ==============================================================================
# Main Application
# ==============================================================================
st.title(t('title'))
st.markdown(f"<p style='text-align: center; color: var(--secondary-text-color);'>{t('subtitle')}</p>", unsafe_allow_html=True)
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)

# Load agents configuration
if st.session_state.agents_cfg is None:
    st.session_state.agents_cfg = load_agents_config()

agents_cfg = st.session_state.agents_cfg
all_agents = agents_cfg.get("agents", [])

# Create tabs
tab_upload, tab_agent_select, tab_pipeline, tab_dashboard, tab_reports = st.tabs([
    f"📂 {t('upload_parse')}",
    f"🛠️ {t('agent_config')}",
    f"🔄 {t('agent_pipeline')}",
    f"📊 {t('dashboard')}",
    f"📤 {t('reports')}"
])

# ==============================================================================
# Tab 1: Upload & Parse
# ==============================================================================
with tab_upload:
    st.header(f"Step 1: {t('upload_parse')}")
    input_method = st.radio("Input method:", ["Upload Files", "Paste Text"], horizontal=True)

    if input_method == "Upload Files":
        files = st.file_uploader("Upload PDF/TXT/MD files", type=["pdf", "txt", "md"], 
                                accept_multiple_files=True, label_visibility="collapsed")
        if files:
            st.session_state.uploaded_docs = [f.name for f in files]
            parsed_chunks = []
            for f in files:
                if f.name.endswith('.pdf'):
                    parsed_chunks.append(parse_pdf_to_text(f.read()))
                else:
                    parsed_chunks.append(parse_plain_file(f.name, f.read()))
            st.session_state.parsed_text = "\n\n---\n\n".join(filter(None, parsed_chunks))
            st.success(f"✅ Processed {len(files)} files. Tokens: {estimate_tokens(st.session_state.parsed_text)}")
    else:
        pasted_text = st.text_area("Paste content:", height=300, placeholder="Paste your text here...")
        if pasted_text:
            st.session_state.parsed_text = pasted_text
            st.success(f"✅ Text loaded. Tokens: {estimate_tokens(pasted_text)}")

    if st.session_state.parsed_text:
        with st.expander("📄 Preview"):
            st.text_area("Parsed Text", st.session_state.parsed_text[:4000], height=200, disabled=True)
        
        # Word cloud visualization
        if st.checkbox("Show Keywords"):
            st.markdown(create_word_cloud_viz(st.session_state.parsed_text), unsafe_allow_html=True)

# ==============================================================================
# Tab 2: Agent Configuration & Selection
# ==============================================================================
with tab_agent_select:
    st.header(f"Step 2: {t('agent_config')}")
    
    if not all_agents:
        st.warning("No agents found in configuration. Please upload an agents.yaml file.")
        
        # Provide download for sample configuration
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "📥 Download Sample agents.yaml",
                data=DEFAULT_SAMPLE_AGENTS.encode('utf-8'),
                file_name="agents_sample.yaml",
                mime="text/yaml",
                use_container_width=True
            )
        with col2:
            st.download_button(
                "📥 Download 31-Agent Config (繁中)",
                data=st.session_state.get("advanced_agents_yaml", "# Upload your own config").encode('utf-8'),
                file_name="agents_fda_advanced_zh.yaml",
                mime="text/yaml",
                use_container_width=True,
                disabled=True,
                help="Full 31-agent configuration (see artifact fda_agents_yaml_zh)"
            )
        
        st.info("Upload a YAML file with the following structure:")
        st.code("""version: 1
agents:
  - id: agent_1
    name: Agent Name
    description: Agent description
    enabled: true
    model:
      provider: gemini
      name: gemini-2.5-flash
      temperature: 0.25
      max_tokens: 4096
    prompt: |
      Your prompt here...
""", language="yaml")
        
        agents_file = st.file_uploader("Upload agents.yaml", type=["yaml", "yml"], key="agents_uploader")
        if agents_file:
            try:
                uploaded_config = yaml.safe_load(agents_file.getvalue())
                
                # Validate structure
                if not isinstance(uploaded_config, dict):
                    st.error("Invalid YAML: Root must be a dictionary")
                elif "agents" not in uploaded_config:
                    st.error("Invalid YAML: Missing 'agents' key")
                elif not isinstance(uploaded_config["agents"], list):
                    st.error("Invalid YAML: 'agents' must be a list")
                else:
                    st.session_state.agents_cfg = uploaded_config
                    st.success(f"✅ Loaded {len(uploaded_config['agents'])} agents from file")
                    st.rerun()
            except yaml.YAMLError as e:
                st.error(f"YAML parsing error: {e}")
            except Exception as e:
                st.error(f"Error loading file: {e}")
    else:
        st.subheader(f"🎯 {t('select_agents')}")
        
        # Display all available agents with selection
        selected_agent_ids = []
        
        for i, agent in enumerate(all_agents):
            agent_id = agent.get("id", f"agent_{i}")
            agent_name = agent.get("name", f"Agent {i+1}")
            agent_desc = agent.get("description", "No description")
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                is_selected = st.checkbox(
                    "Select",
                    value=agent_id in st.session_state.selected_agents,
                    key=f"select_{agent_id}_{i}"  # BUG FIX: Added index 'i' for uniqueness
                )
                if is_selected:
                    selected_agent_ids.append(agent_id)
            
            with col2:
                with st.expander(f"{'✅' if is_selected else '⬜'} {agent_name}", expanded=False):
                    st.markdown(f"**Description:** {agent_desc}")
                    
                    # Agent configuration
                    model_cfg = agent.get("model", {})
                    
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        provider = st.selectbox(
                            "Provider",
                            ["auto", "gemini", "openai", "grok"],
                            index=["auto", "gemini", "openai", "grok"].index(model_cfg.get("provider", "auto")),
                            key=f"provider_{agent_id}_{i}" # BUG FIX: Added index 'i'
                        )
                    with col_b:
                        model_name = st.text_input(
                            "Model",
                            value=model_cfg.get("name", ""),
                            key=f"model_{agent_id}_{i}" # BUG FIX: Added index 'i'
                        )
                    with col_c:
                        temperature = st.slider(
                            "Temperature",
                            0.0, 1.0,
                            float(model_cfg.get("temperature", 0.3)),
                            0.05,
                            key=f"temp_{agent_id}_{i}" # BUG FIX: Added index 'i'
                        )
                    
                    max_tokens = st.number_input(
                        "Max Tokens",
                        min_value=100,
                        max_value=8000,
                        value=int(model_cfg.get("max_tokens", 4096)),
                        step=100,
                        key=f"tokens_{agent_id}_{i}" # BUG FIX: Added index 'i'
                    )
                    
                    prompt = st.text_area(
                        "System Prompt",
                        value=agent.get("prompt", ""),
                        height=200,
                        key=f"prompt_{agent_id}_{i}" # BUG FIX: Added index 'i'
                    )
                    
                    # Update agent configuration
                    agent["model"]["provider"] = provider
                    agent["model"]["name"] = model_name
                    agent["model"]["temperature"] = temperature
                    agent["model"]["max_tokens"] = max_tokens
                    agent["prompt"] = prompt
        
        st.session_state.selected_agents = selected_agent_ids
        
        if selected_agent_ids:
            st.success(f"✅ Selected {len(selected_agent_ids)} agents for pipeline")
        else:
            st.info("Select at least one agent to create a pipeline")

# ==============================================================================
# Tab 3: Agent Pipeline Execution
# ==============================================================================
with tab_pipeline:
    st.header(f"Step 3: {t('agent_pipeline')}")
    
    if not st.session_state.selected_agents:
        st.warning("Please select agents in the Configuration tab first.")
    else:
        st.subheader(f"📋 Pipeline ({len(st.session_state.selected_agents)} agents)")
        
        # Display pipeline visualization
        if PLOTLY_AVAILABLE and st.session_state.pipeline_results:
            fig = create_agent_network_graph(st.session_state.pipeline_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        
        # Execute pipeline step by step
        for idx, agent_id in enumerate(st.session_state.selected_agents):
            agent = next((a for a in all_agents if a.get("id") == agent_id), None)
            if not agent:
                continue
            
            agent_name = agent.get("name", f"Agent {idx+1}")
            
            st.markdown(f"<div class='pipeline-step'>", unsafe_allow_html=True)
            st.markdown(f"### Step {idx+1}: {agent_name}")
            
            # Input section
            col_input, col_output = st.columns(2)
            
            with col_input:
                st.markdown(f"**📥 {t('input')}**")
                
                # Determine input source
                if idx == 0:
                    default_input = st.session_state.parsed_text
                    input_source = "Parsed Document"
                else:
                    prev_agent_id = st.session_state.selected_agents[idx-1]
                    default_input = st.session_state.agent_outputs.get(prev_agent_id, "")
                    prev_agent = next((a for a in all_agents if a.get("id") == prev_agent_id), None)
                    input_source = prev_agent.get("name", "Previous Agent") if prev_agent else "Previous Agent"
                
                st.caption(f"Source: {input_source}")
                
                # Allow modification of input
                agent_input = st.text_area(
                    "Input (editable)",
                    value=default_input,
                    height=200,
                    key=f"input_{agent_id}_{idx}"
                )
            
            with col_output:
                st.markdown(f"**📤 {t('output')}**")
                
                # Display existing output or execution button
                existing_output = st.session_state.agent_outputs.get(agent_id, "")
                
                if existing_output:
                    st.text_area(
                        "Output",
                        value=existing_output,
                        height=200,
                        key=f"output_display_{agent_id}_{idx}",
                        disabled=True
                    )
                else:
                    st.info("Click Execute to run this agent")
            
            # Execution controls
            col_exec, col_modify, col_status = st.columns([2, 2, 3])
            
            with col_exec:
                if st.button(f"▶️ {t('execute')}", key=f"exec_{agent_id}_{idx}"):
                    if not agent_input.strip():
                        st.error("Input is empty!")
                    else:
                        with st.spinner(f"Running {agent_name}..."):
                            output = run_agent(agent, agent_input, clients)
                            st.session_state.agent_outputs[agent_id] = output
                            
                            # Store in pipeline results
                            result = {
                                "step": idx + 1,
                                "agent_id": agent_id,
                                "agent_name": agent_name,
                                "input": agent_input[:500] + "..." if len(agent_input) > 500 else agent_input,
                                "output": output[:500] + "..." if len(output) > 500 else output,
                                "timestamp": time.time()
                            }
                            
                            # Update or append result
                            existing_idx = next((i for i, r in enumerate(st.session_state.pipeline_results) 
                                               if r.get("agent_id") == agent_id), None)
                            if existing_idx is not None:
                                st.session_state.pipeline_results[existing_idx] = result
                            else:
                                st.session_state.pipeline_results.append(result)
                            
                            st.rerun()
            
            with col_modify:
                if existing_output:
                    if st.button(f"✏️ {t('modify')}", key=f"modify_{agent_id}_{idx}"):
                        modified_output = st.text_area(
                            "Edit Output",
                            value=existing_output,
                            height=200,
                            key=f"modify_text_{agent_id}_{idx}"
                        )
                        if st.button("💾 Save", key=f"save_{agent_id}_{idx}"):
                            st.session_state.agent_outputs[agent_id] = modified_output
                            st.success("Output updated!")
                            st.rerun()
            
            with col_status:
                if existing_output:
                    latency = st.session_state.last_latency.get(agent.get("model", {}).get("provider", ""), 0)
                    st.markdown(f"<span class='status-chip chip-pass'>✅ Completed</span>", unsafe_allow_html=True)
                    st.caption(f"⏱️ {latency:.2f}s")
                else:
                    st.markdown(f"<span class='status-chip chip-info'>⏳ Pending</span>", unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add arrow between steps
            if idx < len(st.session_state.selected_agents) - 1:
                st.markdown("<div style='text-align: center; font-size: 2em; color: var(--primary-color);'>⬇️</div>", unsafe_allow_html=True)
        
        # Batch execution
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        if st.button("🚀 Execute All Agents in Sequence", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            current_input = st.session_state.parsed_text
            
            for idx, agent_id in enumerate(st.session_state.selected_agents):
                agent = next((a for a in all_agents if a.get("id") == agent_id), None)
                if not agent:
                    continue
                
                agent_name = agent.get("name", f"Agent {idx+1}")
                status_text.text(f"Running {agent_name}...")
                
                output = run_agent(agent, current_input, clients)
                st.session_state.agent_outputs[agent_id] = output
                
                result = {
                    "step": idx + 1,
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "input": current_input[:500] + "..." if len(current_input) > 500 else current_input,
                    "output": output[:500] + "..." if len(output) > 500 else output,
                    "timestamp": time.time()
                }
                
                existing_idx = next((i for i, r in enumerate(st.session_state.pipeline_results) 
                                   if r.get("agent_id") == agent_id), None)
                if existing_idx is not None:
                    st.session_state.pipeline_results[existing_idx] = result
                else:
                    st.session_state.pipeline_results.append(result)
                
                current_input = output
                progress_bar.progress((idx + 1) / len(st.session_state.selected_agents))
            
            status_text.text("✅ All agents completed!")
            time.sleep(1)
            st.rerun()

# ==============================================================================
# Tab 4: Dashboard
# ==============================================================================
with tab_dashboard:
    st.header(f"📊 {t('dashboard')}")
    
    # KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    
    with k1:
        st.markdown(f"""
            <div class='kpi-card'>
                <h3>Input {t('tokens')}</h3>
                <p>{estimate_tokens(st.session_state.parsed_text)}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with k2:
        total_output = sum(estimate_tokens(output) for output in st.session_state.agent_outputs.values())
        st.markdown(f"""
            <div class='kpi-card'>
                <h3>Output {t('tokens')}</h3>
                <p>{total_output}</p>
            </div>
        """, unsafe_allow_html=True)
    
    with k3:
        completed = len(st.session_state.agent_outputs)
        total = len(st.session_state.selected_agents)
        completion_rate = (completed / total * 100) if total > 0 else 0
        st.markdown(f"""
            <div class='kpi-card'>
                <h3>{t('completion_rate')}</h3>
                <p>{completion_rate:.0f}%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with k4:
        total_latency = sum(st.session_state.last_latency.values())
        st.markdown(f"""
            <div class='kpi-card'>
                <h3>Total {t('latency')}</h3>
                <p>{total_latency:.1f}s</p>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    
    # Visualizations
    col_viz1, col_viz2 = st.columns(2)
    
    with col_viz1:
        st.subheader("⚡ Agent Performance")
        
        if st.session_state.pipeline_results and PLOTLY_AVAILABLE:
            perf_data = []
            for result in st.session_state.pipeline_results:
                agent_name = result.get("agent_name", "Unknown")
                # Estimate processing time (simplified)
                input_tokens = estimate_tokens(result.get("input", ""))
                output_tokens = estimate_tokens(result.get("output", ""))
                
                perf_data.append({
                    "Agent": agent_name,
                    "Input Tokens": input_tokens,
                    "Output Tokens": output_tokens
                })
            
            df_perf = pd.DataFrame(perf_data)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Input', x=df_perf['Agent'], y=df_perf['Input Tokens'], marker_color='#2196F3'))
            fig.add_trace(go.Bar(name='Output', x=df_perf['Agent'], y=df_perf['Output Tokens'], marker_color='#4CAF50'))
            
            fig.update_layout(
                barmode='group',
                template='plotly_dark' if st.session_state.theme_mode == 'dark' else 'plotly_white',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Execute agents to see performance metrics")
    
    with col_viz2:
        st.subheader("🔄 Pipeline Flow")
        
        if st.session_state.pipeline_results and PLOTLY_AVAILABLE:
            fig = create_agent_network_graph(st.session_state.pipeline_results)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Execute agents to see pipeline flow")
    
    # Provider latency breakdown
    if st.session_state.last_latency:
        st.subheader("⏱️ Provider Latency Breakdown")
        
        if PLOTLY_AVAILABLE:
            lat_df = pd.DataFrame(list(st.session_state.last_latency.items()), 
                                 columns=['Provider', 'Latency (s)'])
            
            fig = px.pie(lat_df, values='Latency (s)', names='Provider',
                        color_discrete_sequence=px.colors.sequential.RdBu)
            fig.update_layout(
                template='plotly_dark' if st.session_state.theme_mode == 'dark' else 'plotly_white',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.json(st.session_state.last_latency)
    
    # Word frequency analysis
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.subheader("📝 Content Analysis")
    
    col_w1, col_w2 = st.columns(2)
    
    with col_w1:
        st.markdown("**Input Keywords**")
        if st.session_state.parsed_text:
            st.markdown(create_word_cloud_viz(st.session_state.parsed_text), unsafe_allow_html=True)
    
    with col_w2:
        st.markdown("**Output Keywords**")
        all_outputs = " ".join(st.session_state.agent_outputs.values())
        if all_outputs:
            st.markdown(create_word_cloud_viz(all_outputs), unsafe_allow_html=True)

# ==============================================================================
# Tab 5: Reports & Export
# ==============================================================================
with tab_reports:
    st.header(f"📤 {t('reports')}")
    
    st.subheader("📋 Pipeline Summary")
    
    if st.session_state.pipeline_results:
        # Create summary report
        report_md = f"""# FDA Agent Pipeline Report
## Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
### Configuration
- **Theme:** {st.session_state.theme_style} ({st.session_state.theme_mode} mode)
- **Language:** {st.session_state.language}
- **Total Agents:** {len(st.session_state.selected_agents)}
- **Completed:** {len(st.session_state.agent_outputs)}
### Pipeline Results
"""
        for result in st.session_state.pipeline_results:
            report_md += f"""
#### Step {result['step']}: {result['agent_name']}
**Input Preview:**
```
{result['input'][:300]}...
```
**Output Preview:**
```
{result['output'][:300]}...
```
---
"""
        
        st.markdown(report_md)
        
        # Download options
        st.subheader(f"⬇️ {t('download')} Options")
        
        col_d1, col_d2, col_d3, col_d4 = st.columns(4)
        
        with col_d1:
            st.download_button(
                "📄 Full Report (MD)",
                data=report_md.encode('utf-8'),
                file_name=f"fda_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        with col_d2:
            # Export pipeline results as JSON
            json_export = json.dumps({
                "config": {
                    "theme": st.session_state.theme_style,
                    "mode": st.session_state.theme_mode,
                    "language": st.session_state.language
                },
                "agents": st.session_state.selected_agents,
                "results": st.session_state.pipeline_results,
                "outputs": st.session_state.agent_outputs
            }, ensure_ascii=False, indent=2)
            
            st.download_button(
                "📊 Results (JSON)",
                data=json_export.encode('utf-8'),
                file_name=f"pipeline_results_{int(time.time())}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_d3:
            # Export as CSV
            if st.session_state.pipeline_results:
                df_results = pd.DataFrame(st.session_state.pipeline_results)
                csv_data = df_results.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    "📑 Results (CSV)",
                    data=csv_data,
                    file_name=f"pipeline_results_{int(time.time())}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col_d4:
            # Export configuration
            config_yaml = yaml.dump({
                "agents": st.session_state.agents_cfg.get("agents", []),
                "selected_agents": st.session_state.selected_agents
            }, allow_unicode=True)
            
            st.download_button(
                "⚙️ Config (YAML)",
                data=config_yaml.encode('utf-8'),
                file_name=f"agents_config_{int(time.time())}.yaml",
                mime="text/yaml",
                use_container_width=True
            )
        
        # Results table
        st.subheader("📊 Detailed Results Table")
        if st.session_state.pipeline_results:
            df_display = pd.DataFrame([
                {
                    "Step": r["step"],
                    "Agent": r["agent_name"],
                    "Input Length": len(r["input"]),
                    "Output Length": len(r["output"]),
                    "Timestamp": time.strftime('%H:%M:%S', time.localtime(r["timestamp"]))
                }
                for r in st.session_state.pipeline_results
            ])
            st.dataframe(df_display, use_container_width=True)
    else:
        st.info("Execute agents in the Pipeline tab to generate reports")
        
    # Agent configuration export
    st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
    st.subheader("📝 Export Agent Configuration")
    
    if st.button("Generate agents.yaml"):
        config = {"version": 1, "agents": all_agents}
        yaml_str = yaml.dump(config, allow_unicode=True, default_flow_style=False)
        
        st.code(yaml_str, language="yaml")
        
        st.download_button(
            "⬇️ Download agents.yaml",
            data=yaml_str.encode('utf-8'),
            file_name="agents.yaml",
            mime="text/yaml"
        )

# ==============================================================================
# Footer
# ==============================================================================
st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
emoji = ANIMAL_THEMES[st.session_state.theme_style]["emoji"]
st.markdown(f"""
<div style='text-align: center; padding: 20px; color: var(--secondary-text-color);'>
    <p>{emoji} <strong>Ferrari FDA Agent</strong> | Multi-Agent Pipeline System</p>
    <p style='font-size: 0.9em;'>Theme: {st.session_state.theme_style} | Mode: {st.session_state.theme_mode} | Language: {st.session_state.language}</p>
</div>
""", unsafe_allow_html=True)
