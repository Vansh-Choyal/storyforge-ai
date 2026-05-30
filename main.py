from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict, Literal
from langgraph.graph import START, StateGraph, END
from typing import Annotated, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Send
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A5
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from rich.theme import Theme
from rich.align import Align
from loguru import logger
from dotenv import load_dotenv
from pyfiglet import Figlet
import operator
import random
import re

load_dotenv()

theme = Theme({
    "info":    "bold cyan",
    "success": "bold green",
    "warning": "bold yellow",
    "error":   "bold red",
})

console = Console(theme=theme)

LEVEL_COLORS = {
    "DEBUG":    "bold white",
    "INFO":     "bold cyan",
    "SUCCESS":  "bold green",
    "WARNING":  "bold yellow",
    "ERROR":    "bold red",
    "CRITICAL": "bold white on red",
}

def rich_sink(message):
    record = message.record
    level  = record["level"].name
    color  = LEVEL_COLORS.get(level, "white")
    time   = record["time"].strftime("%H:%M:%S")
    text   = record["message"]

    console.print(
        f"[green]{time}[/] | [{color}]{level:<8}[/] | {text}"
    )

logger.remove()
logger.add(rich_sink, level="DEBUG", colorize=False)
llm_light = ChatOpenAI(model="gpt-4o-mini")
llm = ChatOpenAI(model="gpt-4o")

class Character(BaseModel):
    name:        str = Field(description="Proper fictional name for the character.")
    role:        str = Field(description="The character's role in the story.")
    description: str = Field(description="Detailed information about the character.")

class Characters(BaseModel):
    characters: list[Character]

class Chapter(BaseModel):
    chapter_number: int           = Field(description="Chapter number.")
    name:           str           = Field(description="Chapter title.")
    description:    str           = Field(description="Vivid, detailed, self-contained description of what happens. Must not overlap with other chapters.")
    plot:           Optional[str] = Field(default=None, description="Detailed plot outline, only if needed.")

class Chapters(BaseModel):
    chapters: list[Chapter]

class StoryName(BaseModel):
    name: str = Field(description="A suitable name for the story.")

# Main shared state, passed between different nodes.
class State(TypedDict):
    user_prompt:        str
    genre:              list[Literal['Sci-Fi', 'Mystery', 'Action', 'Drama', 'Horror', 'Romance', 'Thriller', 'Adventure', 'Fantasy']]
    short_summary:      str
    detailed_summary:   str
    characters:         list[Character]
    chapters:           list[Chapter]
    # Chapter generator LLM's work in paralll, so results are merged automatically using `operator.add`.
    completed_chapters: Annotated[list[dict], operator.add]
    book_name:          str
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens:Annotated[int, operator.add]
    total_tokens:       Annotated[int, operator.add]

# Forces LLM to respond in structured Pydantic model.
characters_llm = llm.with_structured_output(Characters, include_raw=True)
chapters_llm   = llm.with_structured_output(Chapters, include_raw=True)
name_llm       = llm.with_structured_output(StoryName, include_raw=True)

# Shared progress bar state used by chapter worker nodes.
_progress: Progress | None = None
_task_id                   = None

# Common workflow progress bar.
workflow_progress = None
workflow_task = None
workflow_percent = 0

def start_workflow_progress():
    global workflow_progress, workflow_task

    workflow_progress = Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    )

    workflow_progress.start()

    workflow_task = workflow_progress.add_task(
        "Generating Book",
        total=100
    )

def update_workflow(percent: int):
    global workflow_progress, workflow_task, workflow_percent
    workflow_percent = percent

    if workflow_progress and workflow_task is not None:
        workflow_progress.update(
            workflow_task,
            completed=percent if percent <=100 else 100
        )
    
    if percent >= 100 and workflow_progress:
        workflow_progress.stop()

# Create a progress bar while chapter workers are running. Removed automatically when finished (transient=True).
def make_progress() -> Progress:
    return Progress(
        SpinnerColumn(style="bold magenta"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="bold green", finished_style="bold green"),
        TextColumn("[bold cyan]{task.completed}[/]/[bold cyan]{task.total}[/] chapters"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    )

# Story titles are LLM-generated, so they cannot be trusted as-is. Removes characters that are invalid in Windows filenames.
def sanitize_filename(name: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "", name).strip()

# Convert generated chapter content into a PDF file.
def generate_pdf(chapters: list[dict], output_file: str) -> None:
    doc = SimpleDocTemplate(
        output_file,
        pagesize=A5,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    label_style = ParagraphStyle(
        "ChapterLabel",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontSize=18,
        leading=18,
        spaceAfter=6,
        textColor="#888888",
    )

    title_style = ParagraphStyle(
        "ChapterTitle",
        parent=styles["Heading1"],
        alignment=TA_CENTER,
        fontSize=22,
        leading=28,
        spaceAfter=30,
    )

    body_style = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontSize=12,
        leading=20,
        firstLineIndent=20,
    )

    elements = []

    for i, ch in enumerate(chapters):

        content = ch["content"]

        # *bold*
        content = re.sub(r"\*(.*?)\*", r"<b>\1</b>", content)

        # _italic_
        content = re.sub(r"_(.*?)_", r"<i>\1</i>", content)

        # `code`
        content = re.sub(
            r"`(.*?)`",
            r'<font face="Courier">\1</font>',
            content
        )

        # Preserve line breaks
        content = content.replace("\n", "<br/>")

        elements += [
            Paragraph(f"Chapter {ch['chapter_number']}", label_style),
            Paragraph(f"<b>{ch['name']}</b>", title_style),
            Spacer(1, 20),
            Paragraph(content, body_style),
        ]

        if i != len(chapters) - 1:
            elements.append(PageBreak())

    doc.build(elements)
    logger.success(f"PDF saved → {output_file}")

def print_total_tokens(total_tokens):
    if total_tokens > 10000:
        logger.warning(
            f"Total [bold yellow]{total_tokens:,}[/] tokens have been used."
        )
    else:
        logger.info(
            f"Total [bold green]{total_tokens:,}[/] tokens have been used."
        )

def gen_short_summary(state: State) -> dict:
    logger.info("Generating short summary.")
    try:
        resp = llm.invoke([
            SystemMessage("Generate a short story summary (~80 words) from the genre and user prompt."),
            HumanMessage(f"Genre: {state['genre']}\nPrompt: {state['user_prompt']}"),
        ])
        usage_metadata = resp.usage_metadata
        logger.success("Short summary done.")
        print_total_tokens(usage_metadata['total_tokens'])

        update_workflow(workflow_percent+12)

        return {"short_summary": resp.content, "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate short summary: {e}")
        raise

def gen_detailed_summary(state: State) -> dict:
    logger.info("Generating detailed summary.")
    try:
        resp = llm.invoke([
            SystemMessage("Generate a detailed story outline (~400 words) from the genre, prompt, and short summary. The opening should feel like the start of a proper book."),
            HumanMessage(f"Genre: {state['genre']}\nPrompt: {state['user_prompt']}\nShort Summary: {state['short_summary']}"),
        ])
        usage_metadata = resp.usage_metadata

        logger.success(f"Detailed summary done ({len(resp.content)} chars).")
        print_total_tokens(state["total_tokens"]+usage_metadata['total_tokens'])
        update_workflow(workflow_percent+18)

        return {"detailed_summary": resp.content, "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate detailed summary: {e}")
        raise

def gen_characters(state: State) -> dict:
    logger.info("Generating characters.")
    try:
        # Genre is optional. Allows character generation to work even if a future workflow omits genre selection.
        genre_text = f"Genre: {', '.join(state['genre'])}\n" if state.get("genre") else ""

        resp = characters_llm.invoke([
            SystemMessage("Generate characters for the story from the summary provided."),
            HumanMessage(f"{genre_text}Summary: {state['short_summary']}"),
        ])
        raw = resp['raw']
        parsed = resp['parsed']

        usage_metadata = raw.usage_metadata
        logger.success(f"Generated {len(parsed.characters)} characters.")
        print_total_tokens(state["total_tokens"]+usage_metadata['total_tokens'])
        update_workflow(workflow_percent+15)


        return {"characters": parsed.characters, "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate characters: {e}")
        raise

def gen_chapters(state: State) -> dict:
    logger.info("Generating chapters.")
    try:
        characters_text = "\n\n".join(
            f"Name: {c.name}\nRole: {c.role}\nDescription: {c.description}"
            for c in state["characters"]
        )
        genre_text = f"Genre: {state['genre']}\n" if state.get("genre") else ""
        resp = chapters_llm.invoke([
            SystemMessage(
                "Generate story chapters from the summaries and characters. "
                "Each chapter needs a title, vivid self-contained description (~200 words), and an optional plot. "
                "Short story → 6–10 chapters. Medium → 10–16. Long → ~20."
            ),
            HumanMessage(
                f"{genre_text}"
                f"Short Summary:\n{state['short_summary']}\n\n"
                f"Detailed Summary:\n{state['detailed_summary']}\n\n"
                f"Characters:\n{characters_text}"
            ),
        ])
        raw = resp['raw']
        parsed = resp['parsed']

        usage_metadata = raw.usage_metadata

        logger.success(f"Generated {len(parsed.chapters)} chapters.")
        print_total_tokens(state["total_tokens"]+usage_metadata['total_tokens'])
        update_workflow(workflow_percent+11)

        return {"chapters": parsed.chapters, "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate chapters: {e}")
        raise

def gen_story_name(state: State) -> dict:
    logger.info("Generating story name.")
    try:
        characters = [f"{c.name}: {c.description}" for c in state["characters"]]
        characters_joined = "\n".join(characters)
        resp = name_llm.invoke([
            SystemMessage("Generate a compelling story name from the summary and characters."),
            HumanMessage(f"Summary:\n{state['short_summary']}\n\nCharacters:\n{characters_joined}"),
        ])
        raw = resp['raw']
        parsed = resp['parsed']

        usage_metadata = raw.usage_metadata

        logger.success(f'Story name: "{parsed.name}"')
        update_workflow(workflow_percent+5)

        return {"book_name": parsed.name, "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate story name: {e}")
        raise


def assign_workers(state: State) -> list[Send]:
    global _progress, _task_id

    _progress = make_progress()
    _task_id  = _progress.add_task("Generating chapters", total=len(state["chapters"]))
    _progress.start()

    logger.info(f"Spawning {len(state['chapters'])} parallel chapter workers.")
    update_workflow(workflow_percent+5)

    # Fan out chapter generation so each chapter can be written independently by a separate LangGraph worker.
    return [
        Send("chapter-content", {
            "chapter_number": ch.chapter_number,
            "name":           ch.name,
            "description":    ch.description,
            "plot":           ch.plot,
            # The final chapter gets a modified prompt so the story ends with a proper conclusion instead of trailing off.
            "is_last":        ch.chapter_number == len(state["chapters"])
        })
        for ch in state["chapters"]
    ]


def gen_chapter(state: dict) -> dict:
    base_prompt = (
        "You are a chapter writer. Write rich, vivid content (~2,000 words) for the given chapter. "
        "Use proper indentation and punctuation. Output story content ONLY — no preamble, chapter "
        "titles, or meta-commentary. The first chapter must feel like the opening of a published book."
    )
    # Give a bit different instructions to the LLM for the last chapter, so the ending feels proper.
    ending_note = " This is the final chapter; give it a satisfying, earned conclusion." if state["is_last"] else ""

    try:
        resp = llm_light.invoke([
            SystemMessage(base_prompt + ending_note),
            HumanMessage(
                f"Chapter {state['chapter_number']}: {state['name']}\n\n"
                f"Description:\n{state['description']}\n"
                + (f"\nPlot:\n{state['plot']}" if state["plot"] else "")
            ),
        ])

        usage_metadata = resp.usage_metadata
        logger.success(f"Chapter {state['chapter_number']} completed: {state['name']}.\nUsed {usage_metadata['total_tokens']} Tokens.")
        update_workflow(workflow_percent+3)

        if _progress and _task_id is not None:
            _progress.advance(_task_id)

        return {"completed_chapters": [{"chapter_number": state["chapter_number"], "name": state["name"], "content": resp.content}], "total_input_tokens": usage_metadata['input_tokens'], "total_output_tokens": usage_metadata['output_tokens'], "total_tokens": usage_metadata['total_tokens']}
    except Exception as e:
        logger.critical(f"Failed to generate chapter {state['chapter_number']} ({state['name']}): {e}")
        raise


def finalize_book(state: State) -> dict:
    if _progress:
        _progress.stop()

    logger.info("Finalizing book.")
    update_workflow(90)

    try:
        # Chapters are generated parallelly, and generation sequence can't be determined.
        chapters_sorted = sorted(state["completed_chapters"], key=lambda c: c["chapter_number"])    # Sorts the chapters by chapter name.
        filename = sanitize_filename(state["book_name"]) + ".pdf"
        generate_pdf(chapters_sorted, filename)
        logger.success("Book generation complete.")
        update_workflow(100)

        return {}
    except Exception as e:
        logger.critical(f"Failed to finalize book: {e}")
        raise

def print_banner(
    content: str | list[str],
    *,
    style: str = "bold cyan",
    center: bool = True,
) -> None:
    """
    Print one or more banner lines using Rich.

    Args:
        content: Single string or list of strings.
        style: Rich style to apply.
        center: Whether to center the content.
    """

    if isinstance(content, str):
        content = [content]

    for item in content:
        if center:
            console.print(Align.center(item), style=style)
        else:
            console.print(item, style=style)

builder = StateGraph(State)

builder.add_node("short-summary",    gen_short_summary)
builder.add_node("detailed-summary", gen_detailed_summary)
builder.add_node("characters",       gen_characters)
builder.add_node("chapters",         gen_chapters)
builder.add_node("story-name",       gen_story_name)
builder.add_node("chapter-content",  gen_chapter)
builder.add_node("chapters-done",    lambda s: {})
builder.add_node("finalize",         finalize_book)

builder.add_edge(START,              "short-summary")
builder.add_edge("short-summary",    "detailed-summary")
builder.add_edge("detailed-summary", "characters")
builder.add_edge("characters",       "chapters")

# Chapter generation and title generation can run independently.
builder.add_conditional_edges("chapters", assign_workers, ["chapter-content"])
builder.add_edge("chapters",         "story-name")

builder.add_edge("chapter-content",  "chapters-done")
builder.add_edge("chapters-done",    "finalize")
builder.add_edge("story-name",       "finalize")
builder.add_edge("finalize",         END)

graph = builder.compile()

if __name__ == "__main__":
    # Generating Banner.
    storyforge = Figlet(font="slant", width=200).renderText("StoryForge")
    ai = Figlet(font="small", width=200).renderText("AI")

    print_banner(
        [
            storyforge,
            ai,
            "[bold white]AI-Powered Book Generation[/]",
            "[dim]LangGraph • OpenAI • ReportLab[/]",
        ]
    )
    
    # Main app starts from here.

    sample_genres = [
        'Sci-Fi',
        'Mystery',
        'Action',
        'Drama',
        'Horror',
        'Romance',
        'Thriller',
        'Adventure',
        'Fantasy',
    ]

    while True:

        prompt = console.input(
            "\n[bold cyan]Story Idea[/] [dim](or 'exit')[/] [bold white]>>> [/]"
        ).strip()

        if prompt.lower() in ("exit", "quit"):
            print_banner("Bye :)", style="bold green")
            break
        if not prompt:
            logger.warning("Please enter a story idea.")
            continue

        genre = console.input(
            "\n[bold magenta]Genres[/][dim](Optional) - Seperated by comma ',' (Sci-Fi, Thriller, Mystery)[/] [bold white]>>> [/]"
        ).strip()

        start_workflow_progress()

        try:
            if genre:
                # Remove whitespace and ignore empty entries.
                genres = [
                    g.strip()
                    for g in genre.split(",")
                    if g.strip()
                ]
            else:
                # If no genre is provided, pick 3 random genres.
                genres = random.sample(sample_genres, k=3)

            logger.info(f"Using genres: {', '.join(genres)}")

            resp = graph.invoke({
                "genre": genres,
                "user_prompt": prompt,
            })

            logger.info(f"Total Input Tokens Used: {resp['total_input_tokens']}")
            logger.info(f"Total Output Tokens Used: {resp['total_output_tokens']}")
            logger.info(f"Total Tokens Used: {resp['total_tokens']}")

        except Exception as e:
            logger.critical(f"Story generation failed: {e}")
            console.print_exception()