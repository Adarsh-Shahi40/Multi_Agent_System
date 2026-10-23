from pathlib import Path
from typing import Any
import json
import re

from agents import create_critic_chain, create_research_agent, create_writer_chain


_TREE_EXCLUDED_NAMES = {".env", ".git", ".venv", "__pycache__"}


def print_project_tree(root: Path | None = None) -> None:
    """Print a compact project tree, excluding environments, secrets, and caches."""
    project_root = (root or Path(__file__).resolve().parent).resolve()
    print(f"{project_root.name}/")

    def print_entries(directory: Path, prefix: str = "") -> None:
        entries = sorted(
            (
                path
                for path in directory.iterdir()
                if path.name not in _TREE_EXCLUDED_NAMES
                and not path.name.endswith(".pyc")
            ),
            key=lambda path: (path.is_file(), path.name.lower()),
        )
        for index, entry in enumerate(entries):
            is_last = index == len(entries) - 1
            connector = "`-- " if is_last else "|-- "
            print(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "|   ")
                print_entries(entry, child_prefix)

    print_entries(project_root)


def _research_tool_outputs(
    messages: list[Any],
) -> tuple[str, list[str], list[dict[str, str]]]:
    search_outputs = [
        message.content
        for message in messages
        if getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "web_search"
        and isinstance(message.content, str)
    ]
    scrape_outputs = [
        message.content
        for message in messages
        if getattr(message, "type", None) == "tool"
        and getattr(message, "name", None) == "scrape_page"
        and isinstance(message.content, str)
    ]

    source_urls: list[str] = []
    related_images: list[dict[str, str]] = []
    clean_search_outputs: list[str] = []
    for output in search_outputs:
        clean_lines = []
        for line in output.splitlines():
            if line.startswith("__TAVILY_IMAGE__:"):
                try:
                    image = json.loads(line.removeprefix("__TAVILY_IMAGE__:"))
                except json.JSONDecodeError:
                    continue
                image_url = image.get("url", "")
                if (
                    isinstance(image_url, str)
                    and image_url.startswith(("https://", "http://"))
                    and image_url not in {item["url"] for item in related_images}
                ):
                    related_images.append(
                        {
                            "url": image_url,
                            "description": str(image.get("description") or "Related image"),
                        }
                    )
                continue
            clean_lines.append(line)

        clean_output = "\n".join(clean_lines).strip()
        clean_search_outputs.append(clean_output)
        for url in re.findall(r"https?://[^\s)<>]+", clean_output):
            cleaned_url = url.rstrip(".,;]")
            if cleaned_url not in source_urls:
                source_urls.append(cleaned_url)

    research_parts = clean_search_outputs + scrape_outputs
    if not research_parts and messages:
        final_message = messages[-1]
        content = getattr(final_message, "content", "")
        if isinstance(content, str) and content:
            research_parts.append(content)

    return "\n\n".join(research_parts), source_urls, related_images[:6]


def _print_stage(message: str) -> None:
    print("\n" + "=" * 72)
    print(message)
    print("=" * 72)
    print(flush=True)


def _response_style(topic: str) -> str:
    """Keep short factual lookups concise while allowing explicitly detailed answers."""
    words = topic.lower().split()
    detail_requests = {
        "analyze", "analysis", "compare", "comparison", "describe", "detail",
        "detailed", "difference", "discuss", "explain", "how", "list", "steps",
        "why",
    }
    if len(words) <= 10 and not detail_requests.intersection(words):
        return (
            "This is a simple factual lookup. Answer directly in 1-2 sentences, "
            "no more than 45 words. Include only the key answer and a brief "
            "qualification if the figure is an estimate or time-sensitive."
        )
    return (
        "Adapt the answer length to the request. Be concise, but include the "
        "explanation, comparison, or details the user explicitly asks for."
    )


def run_research_pipeline(topic: str, show_progress: bool = True) -> dict[str, Any]:
    """Research a topic, draft an answer, critique it, and produce a revision."""
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic cannot be empty.")

    if show_progress:
        _print_stage("STEP 1 - SEARCH AGENT IS WORKING...")

    research_agent = create_research_agent()
    research_response = research_agent.invoke(
        {"messages": [{"role": "user", "content": topic}]}
    )
    research_context, source_urls, related_images = _research_tool_outputs(
        research_response["messages"]
    )
    if show_progress:
        print("\nSEARCH RESULT")
        print("-" * 72)
        print(research_context or "The research agent returned no search results.")
        print(flush=True)

    writer_chain = create_writer_chain()
    critic_chain = create_critic_chain()
    chain_input = {
        "request": topic,
        "research_context": research_context,
        "response_style": _response_style(topic),
    }
    if show_progress:
        _print_stage("STEP 2 - WRITER IS PREPARING A DRAFT...")
    draft = writer_chain.invoke(chain_input)
    if show_progress:
        print("\nDRAFT")
        print("-" * 72)
        print(draft)
        print(flush=True)

    if show_progress:
        _print_stage("STEP 3 - CRITIC IS REVIEWING THE DRAFT...")
    critique = critic_chain.invoke({**chain_input, "draft": draft})
    if show_progress:
        print("\nCRITIC REVIEW")
        print("-" * 72)
        print(critique)
        print(flush=True)

    if show_progress:
        _print_stage("STEP 4 - WRITER IS PREPARING THE FINAL ANSWER...")
    final_answer = writer_chain.invoke(
        {**chain_input, "critic_feedback": critique}
    )

    if show_progress:
        _print_stage("FINAL ANSWER")
        print(final_answer)
        if source_urls:
            print("\nSOURCES")
            for index, url in enumerate(source_urls, start=1):
                print(f"{index}. {url}")
        else:
            print("\nNo source URLs were returned by the search tool.")
        print(flush=True)

    return {
        "topic": topic,
        "research": research_context,
        "sources": source_urls,
        "images": related_images,
        "draft": draft,
        "critique": critique,
        "final_answer": final_answer,
    }


def main() -> None:
    print("Research pipeline ready. Type 'exit' to quit.")
    while True:
        topic = input("\nWhat would you like to research? ").strip()
        if topic.lower() in {"exit", "quit"}:
            break
        if not topic:
            print("Enter a topic or question.")
            continue
        if topic.lower() == "tree":
            print_project_tree()
            continue

        run_research_pipeline(topic)


if __name__ == "__main__":
    main()