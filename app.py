"""Streamlit UI for the multi-agent research system.

Run from inside the Multi_agent_system/ folder:
    python run_app.py
or:
    streamlit run app.py
"""

import time
from typing import Any

import streamlit as st

from agents import create_critic_chain, create_research_agent, create_writer_chain
from pipeline import _research_tool_outputs, _response_style

# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Multi-Agent Research System",
    page_icon="🔎",
    layout="wide",
)

st.markdown(
    """
    <style>
        :root {
            color-scheme: dark;
            --ink: #f5f1ec;
            --muted: #aaa4a0;
            --orange: #ff8a32;
            --orange-bright: #ff5b24;
            --panel: rgba(22, 20, 25, 0.88);
            --line: rgba(255, 255, 255, 0.09);
        }

        .stApp {
            color: var(--ink);
            background:
                radial-gradient(ellipse at 12% 0%, rgba(255, 107, 31, .13), transparent 34%),
                radial-gradient(ellipse at 92% 28%, rgba(119, 70, 139, .11), transparent 32%),
                linear-gradient(145deg, #09090d 0%, #111016 48%, #0a0a0e 100%);
        }
        [data-testid="stHeader"] {background: rgba(9, 9, 13, .72);}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111016 0%, #0b0b10 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div {background: transparent;}
        .block-container {
            max-width: 1280px;
            padding: 2.1rem 3rem 4rem;
        }
        .hero {
            position: relative;
            overflow: hidden;
            padding: 2.7rem 2.8rem 2.4rem;
            border: 1px solid rgba(255, 255, 255, .08);
            border-radius: 26px;
            background:
                radial-gradient(ellipse at 82% 20%, rgba(255, 111, 34, .16), transparent 35%),
                linear-gradient(120deg, rgba(24, 22, 28, .96), rgba(16, 15, 21, .88));
            box-shadow: 0 24px 80px rgba(0, 0, 0, .28);
            margin-bottom: 1.8rem;
        }
        .hero:after {
            content: "";
            position: absolute;
            width: 260px;
            height: 260px;
            right: -90px;
            top: -145px;
            border: 1px solid rgba(255, 138, 50, .23);
            border-radius: 50%;
            box-shadow: 0 0 0 30px rgba(255, 138, 50, .025),
                        0 0 0 65px rgba(255, 138, 50, .02);
        }
        .eyebrow {
            color: var(--orange);
            font: 700 .68rem/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .28em;
            text-transform: uppercase;
        }
        .hero-title {
            margin: .75rem 0 .55rem;
            color: var(--ink);
            font-size: clamp(2.7rem, 6vw, 5rem);
            font-weight: 800;
            letter-spacing: -.065em;
            line-height: 1;
        }
        .hero-title span {
            color: var(--orange);
            text-shadow: 0 0 38px rgba(255, 107, 31, .24);
        }
        .hero-copy {
            max-width: 620px;
            margin: 1rem 0 0;
            color: #b6b0ad;
            font-size: 1rem;
            line-height: 1.75;
        }
        .section-label {
            color: var(--orange);
            font: 700 .68rem/1.4 ui-monospace, SFMono-Regular, Menlo, monospace;
            letter-spacing: .18em;
            text-transform: uppercase;
        }
        .pipeline-title {
            margin: .2rem 0 1rem;
            color: var(--ink);
            font-size: 1.25rem;
            font-weight: 700;
        }
        .pipeline-card {
            display: flex;
            gap: .9rem;
            align-items: flex-start;
            margin: .68rem 0;
            padding: 1rem 1.05rem;
            border: 1px solid var(--line);
            border-radius: 15px;
            background: linear-gradient(120deg, rgba(32, 29, 36, .92), rgba(19, 18, 23, .8));
            box-shadow: 0 8px 24px rgba(0, 0, 0, .12);
        }
        .pipeline-number {
            display: grid;
            width: 2rem;
            height: 2rem;
            flex: 0 0 2rem;
            place-items: center;
            border: 1px solid rgba(255, 138, 50, .32);
            border-radius: 10px;
            color: var(--orange);
            background: rgba(255, 138, 50, .08);
            font: 700 .75rem ui-monospace, monospace;
        }
        .pipeline-card strong {color: #f2eeeb; font-size: .91rem;}
        .pipeline-card p {margin: .23rem 0 0; color: #989198; font-size: .78rem;}
        .stTextInput label, .stTextArea label {color: #d8d1cc !important;}
        .stTextInput input {
            min-height: 3.15rem;
            border: 1px solid rgba(255, 255, 255, .12);
            border-radius: 12px;
            background: rgba(25, 23, 29, .94);
        }
        .stTextInput input:focus {
            border-color: rgba(255, 138, 50, .75);
            box-shadow: 0 0 0 1px rgba(255, 138, 50, .25);
        }
        .stButton > button, .stDownloadButton > button {
            border: 1px solid rgba(255, 255, 255, .1);
            border-radius: 11px;
            color: #eee8e3;
            background: rgba(255, 255, 255, .045);
            transition: transform .18s ease, border-color .18s ease, background .18s ease;
        }
        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-1px);
            border-color: rgba(255, 138, 50, .6);
            color: white;
            background: rgba(255, 138, 50, .1);
        }
        .stFormSubmitButton > button {
            min-height: 3.15rem;
            border: 0;
            border-radius: 12px;
            color: #1a0d07;
            background: linear-gradient(100deg, #ff9a3c, #ff5a24);
            box-shadow: 0 8px 28px rgba(255, 93, 34, .24);
            font-weight: 800;
        }
        .stFormSubmitButton > button:hover {
            border: 0;
            color: #160a05;
            background: linear-gradient(100deg, #ffad55, #ff7139);
            box-shadow: 0 10px 34px rgba(255, 93, 34, .34);
        }
        [data-testid="stTabs"] button {color: #aaa4a0;}
        [data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--orange);
            border-bottom-color: var(--orange);
        }
        [data-testid="stMetric"] {
            padding: .9rem;
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
        }
        [data-testid="stImage"] img {
            height: 190px;
            border: 1px solid var(--line);
            border-radius: 14px;
            object-fit: cover;
        }
        [data-testid="stImage"] {
            padding: .45rem;
            border: 1px solid var(--line);
            border-radius: 17px;
            background: var(--panel);
        }
        hr {border-color: var(--line);}
        @media (max-width: 700px) {
            .block-container {padding: 1.1rem 1rem 3rem;}
            .hero {padding: 2rem 1.4rem; border-radius: 20px;}
            .hero-title {font-size: 2.8rem;}
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []  # list of result dicts
if "selected" not in st.session_state:
    st.session_state.selected = None  # index into history
if "topic_input" not in st.session_state:
    st.session_state.topic_input = ""


# --------------------------------------------------------------------------- #
# Pipeline runner (same steps as pipeline.run_research_pipeline, but with
# live progress reporting so the UI can show each stage as it happens)
# --------------------------------------------------------------------------- #
def run_with_progress(topic: str) -> dict[str, Any]:
    timings: dict[str, float] = {}

    with st.status("Running research pipeline...", expanded=True) as status:
        # Step 1 - research agent
        st.write("🔎 **Step 1/4** - Search agent is gathering information...")
        t0 = time.perf_counter()
        research_agent = create_research_agent()
        research_response = research_agent.invoke(
            {"messages": [{"role": "user", "content": topic}]}
        )
        research_context, source_urls, related_images = _research_tool_outputs(
            research_response["messages"]
        )
        timings["Research"] = time.perf_counter() - t0

        writer_chain = create_writer_chain()
        critic_chain = create_critic_chain()
        chain_input = {
            "request": topic,
            "research_context": research_context,
            "response_style": _response_style(topic),
        }

        # Step 2 - draft
        st.write("✍️ **Step 2/4** - Writer is preparing a draft...")
        t0 = time.perf_counter()
        draft = writer_chain.invoke(chain_input)
        timings["Draft"] = time.perf_counter() - t0

        # Step 3 - critique
        st.write("🧐 **Step 3/4** - Critic is reviewing the draft...")
        t0 = time.perf_counter()
        critique = critic_chain.invoke({**chain_input, "draft": draft})
        timings["Critique"] = time.perf_counter() - t0

        # Step 4 - final answer
        st.write("✅ **Step 4/4** - Writer is preparing the final answer...")
        t0 = time.perf_counter()
        final_answer = writer_chain.invoke(
            {**chain_input, "critic_feedback": critique}
        )
        timings["Final"] = time.perf_counter() - t0

        status.update(label="Research complete", state="complete", expanded=False)

    return {
        "topic": topic,
        "research": research_context,
        "sources": source_urls,
        "images": related_images,
        "draft": draft,
        "critique": critique,
        "final_answer": final_answer,
        "timings": timings,
    }


def render_result(result: dict[str, Any]) -> None:
    st.subheader(result["topic"])

    tab_final, tab_research, tab_draft, tab_critic = st.tabs(
        ["✅ Final Answer", "🔎 Research", "✍️ Draft", "🧐 Critic Review"]
    )

    with tab_final:
        st.markdown(result["final_answer"])

        if result.get("images"):
            st.markdown("##### Related images")
            st.caption("Images returned by Tavily for this topic.")
            image_columns = st.columns(3, gap="medium")
            for index, image in enumerate(result["images"]):
                with image_columns[index % len(image_columns)]:
                    st.image(
                        image["url"],
                        caption=image["description"],
                        use_container_width=True,
                    )
        else:
            st.caption("No related images were returned for this topic.")

        st.markdown("##### Sources")
        if result["sources"]:
            for i, url in enumerate(result["sources"], start=1):
                st.markdown(f"{i}. [{url}]({url})")
        else:
            st.caption("No source URLs were returned by the search tool.")

        st.download_button(
            "⬇️ Download answer (.md)",
            data=_to_markdown(result),
            file_name="research_answer.md",
            mime="text/markdown",
        )

    with tab_research:
        st.caption("Raw output collected by the search agent")
        st.text_area(
            "Research context",
            result["research"] or "The research agent returned no search results.",
            height=400,
            label_visibility="collapsed",
        )

    with tab_draft:
        st.markdown(result["draft"])

    with tab_critic:
        st.markdown(result["critique"])

    timings = result.get("timings")
    if timings:
        cols = st.columns(len(timings))
        for col, (name, secs) in zip(cols, timings.items()):
            col.metric(name, f"{secs:.1f}s")


def _to_markdown(result: dict[str, Any]) -> str:
    sources = "\n".join(
        f"{i}. {u}" for i, u in enumerate(result["sources"], start=1)
    ) or "No sources returned."
    return (
        f"# {result['topic']}\n\n"
        f"{result['final_answer']}\n\n"
        f"## Sources\n{sources}\n"
    )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🔎 Research System")
    st.caption("Search agent → Writer → Critic → Final answer")

    st.markdown("**Examples**")
    examples = [
        "Population of India in 2025",
        "Explain how large language models are trained",
        "Compare solar and wind energy costs",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.topic_input = ex

    st.divider()
    st.markdown("**History**")
    if not st.session_state.history:
        st.caption("No searches yet.")
    else:
        for i in range(len(st.session_state.history) - 1, -1, -1):
            item = st.session_state.history[i]
            label = item["topic"][:40] + ("…" if len(item["topic"]) > 40 else "")
            if st.button(label, key=f"hist_{i}", use_container_width=True):
                st.session_state.selected = i
        if st.button("🗑️ Clear history", use_container_width=True):
            st.session_state.history = []
            st.session_state.selected = None
            st.rerun()

# --------------------------------------------------------------------------- #
# Main area
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Multi-agent AI research system</div>
        <div class="hero-title">Research<span>Mind</span></div>
        <p class="hero-copy">
            Four specialized AI agents collaborate — searching, reading, writing,
            and critiquing — to turn your question into a polished, source-backed answer.
        </p>
    </section>
    """,
    unsafe_allow_html=True,
)

input_col, pipeline_col = st.columns([1.08, 0.92], gap="large")
with input_col:
    st.markdown('<div class="section-label">Start exploring</div>', unsafe_allow_html=True)
    st.markdown("### What are you curious about?")
    st.caption("Ask a quick question or request a deeper research report.")

    with st.form("research_form", clear_on_submit=False):
        topic = st.text_input(
            "Research topic",
            key="topic_input",
            placeholder="e.g. Quantum computing breakthroughs in 2025",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button(
            "✦  Run research pipeline",
            type="primary",
            use_container_width=True,
        )

    st.markdown('<div class="section-label" style="margin-top:1.35rem">Try a prompt</div>', unsafe_allow_html=True)
    prompt_cols = st.columns(2)
    prompt_examples = [
        "BCCI net worth",
        "Explain how LLMs are trained",
        "Compare solar and wind energy",
        "Latest space exploration news",
    ]
    for index, example in enumerate(prompt_examples):
        if prompt_cols[index % 2].button(
            example,
            key=f"prompt_{index}",
            use_container_width=True,
        ):
            st.session_state.topic_input = example
            st.rerun()

with pipeline_col:
    st.markdown('<div class="section-label">Behind every answer</div>', unsafe_allow_html=True)
    st.markdown('<div class="pipeline-title">Your research pipeline</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="pipeline-card">
            <div class="pipeline-number">01</div>
            <div><strong>Search agent</strong><p>Finds current information and credible sources.</p></div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-number">02</div>
            <div><strong>Reader agent</strong><p>Opens useful pages and extracts readable content.</p></div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-number">03</div>
            <div><strong>Writer agent</strong><p>Builds a clear answer shaped around your question.</p></div>
        </div>
        <div class="pipeline-card">
            <div class="pipeline-number">04</div>
            <div><strong>Critic agent</strong><p>Checks relevance, accuracy, and clarity before delivery.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if submitted:
    if not topic.strip():
        st.warning("Enter a topic or question.")
    else:
        try:
            result = run_with_progress(topic.strip())
            st.session_state.history.append(result)
            st.session_state.selected = len(st.session_state.history) - 1
        except Exception as exc:  # noqa: BLE001
            st.error(f"Something went wrong while running the pipeline: {exc}")

selected = st.session_state.selected
if selected is not None and selected < len(st.session_state.history):
    st.divider()
    render_result(st.session_state.history[selected])