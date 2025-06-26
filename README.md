# 📚 Automated Book Publications Workflow

A modular pipeline to rewrite public domain books using AI, enable human review, and manage versioning with ChromaDB and reinforcement learning feedback.

---

## 📦 Features

- 📖 Web scraping of books from sources like Wikisource
- ✍️ AI rewriting of content using LLMs
- ✅ AI-powered grammar and quality checks
- 🧠 Human-in-the-loop review via Streamlit
- 🧾 Content versioning and retrieval with ChromaDB
- 🤖 Reinforcement Learning loop using PPO
- 🔄 Exportable rewritten content for digital publication

---

## 📁 Project Structure

<pre>
Automated_book_publications_workflow/
├── AI_pipeline/
│   ├── ai_writer.py            # Rewriting with LLM
│   ├── ai_reviewer.py          # Grammar/quality checker
│   └── human_interface.py      # Feedback mechanism (Streamlit, etc.)
├── chromadb/                   # ChromaDB vector store (persisted files)
├── data/                       # Raw, rewritten, and approved content
├── human_feedback/             # Human ratings/comments logs
├── scraping/
│   ├── scraper.py              # Scrape Wikisource or other text
│   └── screenshot.py           # (Optional) Visual snapshot
├── utils/                      # Common utility functions
├── versioning/
│   ├── chroma_handler.py       # Version management using ChromaDB
│   └── rl_search.py            # Reinforcement learning-based retrieval
├── main.py                     # Pipeline entry point
|-req.txt    
├── ppo_trainer.py              # PPO trainer for feedback learning
├── PPO_reranker.py
├──ppo_model.pt
├── Revied_verions.txt        #outptus of PPO predtictions    
├── run_review_streamlit.py     # Run Streamlit interface
└── README.md                   # This file
</pre>
---

## Component-wise Explanation

This workflow is modular, with each component playing a critical role in the automated publication pipeline:

### 1. 🔍 Scraping & Screenshots (`scraping/`)

* **`scraper.py`**: Utilizes `Playwright` to efficiently extract raw HTML and text content from specified Wikisource pages, serving as the initial content source.
* **`screenshot.py`**: Captures full-page screenshots as PNG files. This is useful for archival purposes, visual verification, or as a form of visual version control for the source material.

### 2. ✍️ AI Writing & Review (`ai_pipeline/`)

This module leverages powerful Large Language Models (LLMs) to transform and refine content.

* **`ai_writer.py`**: Takes raw chapters as input and employs an LLM (specifically, **Phi-2 LLM**) to "spin" or rewrite the content, adapting it to a new style, tone, or specific requirements.
* **`ai_reviewer.py`**: Performs an additional layer of AI-driven refinement, focusing on grammar correction, style enhancements, factual consistency checks, and overall content improvement on the AI-written output. This also utilizes the **Phi-2 LLM**.
* **`human_interface.py`**: Provides a crucial bridge for human interaction. It allows human reviewers to interact with the content, offering functionalities to approve, edit, or add comments. This interface can be accessed via command-line prompts or an optional web UI built with Streamlit.

### 3. 🧑‍💼 Human-in-the-Loop

A core principle of this workflow is the integration of human intelligence at strategic points to simulate multi-stage editorial processes. This ensures high-quality output and maintains human oversight:

* **Writer AI ➝ Human Writer ➝ Reviewer AI ➝ Human Reviewer ➝ Editor**
    This controlled flow, primarily managed via `human_interface.py`, ensures that AI-generated content is vetted and refined by human experts at critical junctures.

### 4. 📡 Agentic API Flow

The entire workflow operates on an agentic model where each module seamlessly passes data to the next, maintaining a standardized data structure (typically `dict` or JSON) for smooth integration.

* **`main.py`**: This is the orchestrator script that manages the entire pipeline, coordinating calls from content scraping, through AI writing and reviewing, and finally to content saving and versioning.

### 5. 🗂️ Versioning & Consistency (`versioning/`)

Robust version control is essential for tracking content evolution and facilitating content retrieval.

* **`chroma_handler.py`**: This component is responsible for storing every version of content (raw, spun by AI, human-reviewed, final approved) within `ChromaDB`. Each content version is saved as a document accompanied by rich metadata such as chapter ID, version number, and timestamp.
* **`rl_search.py`**: Implements a reinforcement learning (RL)-based search algorithm. This advanced search capability allows for the intelligent retrieval of the most relevant or "best" version of content based on specific user queries, moving beyond simple keyword matching.

---

## Core Tools

This project leverages the following key technologies to achieve its automated workflow:

* **Python**: The primary development language for the entire system.
* **Playwright**: Utilized for reliable web scraping and capturing full-page screenshots.
* **LLM (Large Language Models)**: The AI engine for content "spinning" (rewriting), detailed reviewing, and enhancement. **Notably, Phi-2 LLM is used to avoid external API requests, allowing for local inference.**
* **ChromaDB**: Serves as the vector database for efficient content versioning and storage.
* **RL Search Algorithm**: Powers the intelligent and consistent retrieval of published content from ChromaDB.

---

## Setup and Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/DASIREDDYCHAITANYA/Automated_book_publications_workflow.git](https://github.com/DASIREDDYCHAITANYA/Automated_book_publications_workflow.git)
    cd Automated_book_publications_workflow
    ```
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    *(Note: A `requirements.txt` file is typically included in Python projects for exact dependencies. Assuming one exists, install it as follows)*
    ```bash
    pip install -r requirements.txt
    ```
    *If `requirements.txt` is not provided, you would need to install individual libraries like `playwright`, `chromadb`, `transformers` (for LLM interactions if not using direct API calls), `streamlit`, etc.*

4.  **Set up API Keys:**
    * If using any external APIs (e.g., for certain LLM functionalities not handled by Phi-2), you will need corresponding API keys. Store these securely, ideally as environment variables.
    * **For Phi-2 LLM**: Ensure you have the necessary model files and local inference setup configured as per Phi-2's documentation.

5.  **Install Playwright browser executables:**
    ```bash
    playwright install
    ```

---

## Usage: Chapter Processing Workflow

The `main.py` script orchestrates the entire workflow, demonstrating a seamless process for publishing a single chapter, for example, "The Gates of Morning/Book 1/Chapter 1" from Wikisource:

1.  **Scraping & Screenshots**: The system first fetches the content and saves full-page screenshots from the specified Wikisource URL (e.g., `https://en.wikisource.org/wiki/The_Gates_of_Morning/Book_1/Chapter_1`). This step uses Playwright to accurately extract the raw text and capture visual records.
2.  **AI Writing & Review**: The scraped chapter is then fed into the AI pipeline. An **AI Writer** (powered by **Phi-2 LLM**) "spins" or rewrites the chapter into a desired style or tone. Subsequently, an **AI Reviewer** (also using **Phi-2 LLM**) refines this output for grammar, style, and content quality.
3.  **Human-in-the-Loop**: The processed content enters a crucial multi-stage human review. This involves **multiple iterations with human input** from writers, reviewers, and editors, ensuring the content meets human quality standards before finalization.
4.  **Agentic API**: Throughout these stages, the **Agentic API** ensures a seamless and standardized content flow between the various AI agents and human interaction points, maintaining data integrity and pipeline efficiency.
5.  **Versioning & Consistency**: Once the chapter is finalized through all AI and human review stages, its **final version is saved**. This content, along with its metadata, is stored in **ChromaDB**. A **Reinforcement Learning (RL) Search Algorithm** is then used to enable consistent and intelligent retrieval of this published content, ensuring that the best and most relevant version is always accessible.

To initiate the workflow:

```bash
python main.py

