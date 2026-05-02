# Aegis Breaker: Advanced LLM Red-Teaming Framework

Aegis Breaker is an asynchronous red-teaming framework designed to probe LLM-based applications for vulnerabilities. It automates the generation, execution, and evaluation of adversarial payloads across multiple attack vectors, providing real-time telemetry and detailed breach reports.

### Core Capabilities
*   **Multi-Vector Orchestration:** Simultaneous execution of Direct Injection, RAG Injection, and Tool Abuse simulations.
*   **SmartMutator Engine:** Real-time, LLM-driven payload evolution that reinforces successful bypasses to discover complex exploits.
*   **Asynchronous Scalability:** A distributed worker architecture managed by a centralized Master Orchestrator for high-throughput testing.
*   **Heuristic Scoring:** A sophisticated evaluation engine that determines compromise confidence and categorizes attack severity.
*   **Live Telemetry:** A dedicated CLI Dashboard providing real-time PPS (Requests per Second), success rates, and worker health monitoring.

## Getting Started

### 1. Prerequisites
*   **Python 3.10+**
*   An active **ngrok** account (if testing via public tunnels).
*   An **API Key** for Google Gemini (optional, if targeting production models).

### 2. Installation
Clone the repository and install the required dependencies:
```bash
pip install aiohttp fastapi uvicorn pyngrok pydantic
```

### 3. Environment Configuration
The framework relies on environment variables for targeting and authentication. Set these in your terminal or a `.env` file:
*   `TARGET_API_KEY`: Your Gemini API key or a placeholder for the Mock API.
*   `TARGET_API_URL`: The endpoint of the target LLM (e.g., your ngrok URL or `http://localhost:8001/chat`).
*   `TARGET_USER_ID`: A unique identifier for the attack session (default: `red-team-1`).

### 4. Running the Mock API (Safe Testing)
Before launching a full campaign, it is recommended to run the local Mock API to verify the orchestration logic without incurring costs or hitting rate limits:
```bash
python MOCK_API.py
```
*The Mock API will start on port 8001 and provide a public ngrok URL if configured.*

### 5. Launching the Attack Dashboard
The primary interface for Aegis Breaker is the CLI Dashboard. This orchestrates the `integrated_orchestrator.py` and provides live telemetry.
```bash
python CLI/dashboard.py
```

## Usage Guide
Once the dashboard is active, you can select from several attack modes:

1.  **Direct Injection:** Attempts to bypass system prompts using the `BasePrompts.txt` library.
2.  **RAG Injection:** Simulates retrieval-layer poisoning using the `rag_injection_framework.py`.
3.  **Tool Abuse:** Probes for unauthorized access to internal functions (e.g., file readers, DB lookups).
4.  **Combined Sprint:** Executes all attack vectors simultaneously with high-concurrency workers.

### Monitoring Results
*   **Live Metrics:** View PPS, Success Rate, and Worker Health directly in the terminal.
*   **Detailed Logs:** Check `rag_test_documents.json` for generated RAG artifacts.
*   **Telemetry Data:** Technical metrics are persisted in `CLI/metrics.json`.

### Framework Components

- **`integrated_orchestrator.py`**: The central delivery engine that coordinates workers and rate-limiting.
- **`LLM_Client_Adapter.py`**: A robust abstraction layer for communicating with Gemini or Mock APIs.
- **`success_eval_logic.py`**: The heuristic engine used to score and categorize model responses.
- **`CLI/dashboard.py`**: The real-time monitoring and control interface.
- **`rag_injection_framework.py`**: Specialized logic for generating and simulating RAG-layer attacks.
- **`MOCK_API.py`**: A FastAPI-based server for isolated, deterministic testing.

## Project Structure
```text
.
├── CLI/                # Dashboard and metrics persistence
├── engine/             # Core orchestration logic
├── payloads/           # Attack libraries (Prompt, RAG, Tool)
└── MOCK_API.py         # Local simulation environment
```
