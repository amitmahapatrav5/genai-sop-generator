# README

This project analyzes raw HTML files and automatically extracts **interactive actions** and **read-only information** as a human would perceive them in a rendered web page. It uses an LLM-powered backend to semantically understand UI elements and a lightweight frontend to upload files and visualize results.


**Flow**
- Upload an HTML file through a web interface
- Backend parses visible UI elements using an LLM
- Features are categorized into:
  - **Actions**: Interactive user flows (forms, buttons, navigations)
  - **Information**: Static, read-only content grouped into meaningful summaries
- Results are displayed in a clean, user-friendly UI

**Stack**
- Python
- Ollama
- gpt-oss-120b
- UV
- FastAPI
- Uvicorn
- HTML
- Tailwind CSS