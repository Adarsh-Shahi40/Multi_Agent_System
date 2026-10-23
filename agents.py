import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_groq import ChatGroq

from tools import scrape_page, web_search

load_dotenv()


def _create_groq_model() -> ChatGroq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in the .env file.")

    return ChatGroq(
        api_key=api_key,
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
    )


def create_writer_chain() -> Runnable[dict[str, str], str]:
    """Create an LCEL chain that writes a clear response from a request and context."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a careful writer. Write a clear, accurate, well-organized "
                "response that addresses the user's request. Use the supplied context "
                "when available, and do not invent facts. If editor feedback is "
                "provided, revise the draft to address it while preserving accurate "
                "details and answering the original request. Follow the requested "
                "response length exactly. For a simple factual lookup, give the "
                "direct answer in 1-2 sentences and at most 45 words. Do not add "
                "background, headings, or a conclusion unless requested.",
            ),
            (
                "human",
                "Request:\n{request}\n\n"
                "Research context (may be empty):\n{research_context}\n\n"
                "Response length:\n{response_style}\n\n"
                "Editor feedback (may be empty):\n{critic_feedback}",
            ),
        ]
    )
    prepare_input = RunnableLambda(
        lambda values: {
            "request": values["request"],
            "research_context": values.get("research_context", ""),
            "response_style": values.get(
                "response_style",
                "Use the amount of detail requested by the user.",
            ),
            "critic_feedback": values.get("critic_feedback", ""),
        }
    )
    return prepare_input | prompt | _create_groq_model() | StrOutputParser()


def create_critic_chain() -> Runnable[dict[str, str], str]:
    """Create an LCEL chain that critiques a draft and suggests concrete improvements."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a constructive, rigorous editor. Evaluate the draft for "
                "accuracy against the provided context, relevance to the request, "
                "clarity, completeness, requested length, and unsupported claims. "
                "For simple factual lookups, require a direct 1-2 sentence answer "
                "of at most 45 words. Never recommend extra background or expansion "
                "unless the user asked for it. Return concise, actionable feedback. "
                "If the draft is strong, say so and identify any remaining caveats.",
            ),
            (
                "human",
                "Original request:\n{request}\n\n"
                "Requested response length:\n{response_style}\n\n"
                "Available context:\n{research_context}\n\n"
                "Draft to critique:\n{draft}",
            ),
        ]
    )
    prepare_input = RunnableLambda(
        lambda values: {
            "request": values["request"],
            "research_context": values.get("research_context", ""),
            "response_style": values.get(
                "response_style",
                "Use the amount of detail requested by the user.",
            ),
            "draft": values["draft"],
        }
    )
    return prepare_input | prompt | _create_groq_model() | StrOutputParser()


def create_research_agent():
    """Create a Groq agent that can search the web and scrape public webpages."""
    return create_agent(
        model=_create_groq_model(),
        tools=[web_search, scrape_page],
        system_prompt=(
            "You are a helpful research assistant. Use web_search to find current "
            "information. When a search result URL has useful detail, use scrape_page "
            "to read the page. If scraping is blocked or unavailable, continue using "
            "the Tavily search result snippets or try another result URL. Base answers "
            "on retrieved content and include source URLs when available. Explain "
            "when information could not be verified rather than inventing results."
        ),
    )


def main() -> None:
    agent = create_research_agent()
    print("Research agent ready. Type 'exit' to quit.")

    while True:
        query = input("\nWhat would you like to research? ").strip()
        if query.lower() in {"exit", "quit"}:
            break
        if not query:
            print("Enter a topic or question.")
            continue

        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        print(response["messages"][-1].content)


if __name__ == "__main__":
    main()