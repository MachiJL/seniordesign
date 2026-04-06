**Overview**

The CLI Dashboard is a live monitoring and control interface for the Red Team Attack Framework. It allows users to launch automated LLM security attack simulations, observe runtime metrics, and review final attack summaries without modifying source code.

The dashboard communicates with the orchestrator through a shared metrics.json file and provides both real-time updates and final reporting once an attack sprint completes.

This component is designed for:

Prompt injection testing
Tool abuse simulation
RAG injection testing
Combined multi-vector attack campaigns
Mutation-driven exploit discovery monitoring
Features

The dashboard provides:

• Real-time attack monitoring
• Attack-mode selection interface
• Metrics refresh controls
• Continuous live telemetry mode
• Attack lifecycle management
• Worker health monitoring
• Success-rate visualization
• Latency tracking
• Attack-type distribution reporting
• Severity distribution reporting

**Folder Structure Requirement**

The dashboard expects the following directory layout:

project_root/
│
├── intergrated_orchestrator.py
├── LLM_Client_Adapter.py
├── success_eval_logic.py
│
└── CLI/
    ├── dashboard.py
    └── metrics.json (auto-generated)

The orchestrator must exist at the project root level.

How the Dashboard Works

The dashboard reads runtime telemetry from:

CLI/metrics.json

This file is automatically updated by:

intergrated_orchestrator.py

Metrics refresh every 2 seconds in continuous mode.

Launching the Dashboard

From inside the CLI directory:

python dashboard.py

Or from the project root:

python CLI/dashboard.py
Dashboard Menu Options

When started, the dashboard presents the following options:

1) Start prompt injection attack
2) Start RAG injection attack
3) Start tool abuse attack
4) Start combined attack sprint
5) Stop running attack
6) Reset metrics
7) Refresh display
8) Continuous refresh mode
9) Exit
Attack Modes Explained
Prompt Injection Mode

Tests jailbreak attempts such as:

Ignore all instructions
Reveal system prompt
Enter developer mode
RAG Injection Mode

Tests retrieval-layer manipulation attacks such as:

Inject malicious document context
Override retrieval instructions
Leak embedded knowledge-base secrets

Requires RAG simulation support enabled in orchestrator.

Tool Abuse Mode

Tests improper tool execution exposure such as:

List available tools
Access environment variables
Query database credentials
Read filesystem contents

Uses simulated tool execution responses via adapter.

Combined Attack Sprint Mode

Runs all attack types simultaneously:

prompt injection
tool abuse
rag injection
mutation expansion
recursive exploit discovery

This mode produces the most comprehensive security evaluation.

Metrics Displayed During Execution

While running:

Runtime
Requests/sec (PPS)
Total Requests Sent
Successful Bypasses
Errors
Success Rate
Average Latency
Last Event Triggered

After completion:

Target Model
Total Tests Run
Successful Bypasses
Success Rate
Average Confidence Score
Average API Latency
Worker Health
Attack Type Distribution
Severity Distribution
Continuous Refresh Mode

Selecting option:

8) Continuous refresh mode

Enables live telemetry updates every:

2 seconds

Exit using:

CTRL + C
Metrics Reset Option

Selecting:

6) Reset metrics

Deletes:

CLI/metrics.json

This clears all dashboard data and prepares the system for a fresh run.

Worker Health Indicator

Example:

Worker Health : 5/5

Means:

5 workers active
0 workers failed

If workers fail due to adapter errors or API issues, this value decreases.

Orchestrator Integration

The dashboard launches:

intergrated_orchestrator.py

Using subprocess execution with environment variables:

ATTACK_MODE
LAUNCH_DASHBOARD

This prevents recursive dashboard spawning.

Live Status Indicators

Dashboard status messages include:

RUNNING
COMPLETED
WAITING FOR METRICS DATA

These reflect orchestrator lifecycle state.

Troubleshooting
Issue: Orchestrator not found

Error:

Could not find intergrated_orchestrator.py

Solution:

Ensure file exists at:

project_root/intergrated_orchestrator.py
Issue: No metrics displayed

Cause:

metrics.json not generated yet

Solution:

Start an attack sprint first.

Issue: Dashboard exits immediately

Cause:

Python path mismatch

Solution:

Run dashboard using the same interpreter as orchestrator.

Example:

python CLI/dashboard.py
Issue: Attack will not start

Cause:

Existing orchestrator instance already running

Solution:

Stop running attack first:

Option 5

Then restart.

Safe Shutdown Behavior

If dashboard exits while attack is running:

orchestrator process terminates automatically

This prevents orphan worker threads.

Intended Usage Context

This dashboard is designed for:

LLM security testing
prompt injection research
tool abuse simulation
RAG attack experimentation
mutation-based fuzzing visualization

It is safe for offline mock testing environments and controlled evaluation scenarios.

Recommended Workflow

Typical workflow:

Start dashboard
Select attack mode
Observe live telemetry
Switch to continuous refresh
Review final summary
Reset metrics
Run next experiment
Version Compatibility

Compatible with:

CompoundMaster Orchestrator
SmartMutator Engine
LLM Adapter (mock + Gemini modes)
SuccessEvaluator scoring system
