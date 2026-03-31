import time
import json
import os
import subprocess

import sys

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")


def load_metrics():
    if not os.path.exists(METRICS_FILE):
        return None
    try:
        with open(METRICS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def clear():
    os.system("cls" if os.name == "nt" else "clear")


def _print_metrics(start_time):
    metrics = load_metrics()
    runtime = time.time() - start_time

    print("═" * 60)
    print("      RED TEAM ATTACK FRAMEWORK – LIVE DASHBOARD")
    print("═" * 60)

    if not metrics:
        print("Status: Waiting for metrics data...")
    else:
        total = metrics.get("total_sent", 0)
        success = metrics.get("success", 0)
        errors = metrics.get("errors", 0)
        pps = metrics.get("pps", 0)
        avg_latency = metrics.get("avg_latency_ms", 0)

        success_rate = (success / total * 100) if total > 0 else 0

        print(f"Status: RUNNING")
        print(f"Runtime: {runtime:.1f} seconds")
        print("-" * 60)
        print(f"Requests Per Second (PPS): {pps}")
        print(f"Total Requests Sent:       {total}")
        print(f"Successful Responses:      {success}")
        print(f"Errors:                    {errors}")
        print(f"Success Rate:              {success_rate:.2f}%")
        print(f"Average Latency:           {avg_latency:.2f} ms")
        print("-" * 60)

        if "last_event" in metrics:
            print(f"Last Event: {metrics['last_event']}")

    print("═" * 60)


def run_orchestrator(root_dir, target_url=None, api_key=None):
    python_exe = sys.executable
    orchestrator_path = os.path.join(root_dir, "intergrated_orchestrator.py")
    env = os.environ.copy()
    # tell the orchestrator NOT to spawn a dashboard (we are the dashboard)
    env["LAUNCH_DASHBOARD"] = "0"
    if target_url:
        env["TARGET_API_URL"] = target_url
    if api_key:
        env["TARGET_API_KEY"] = api_key
        
    proc = subprocess.Popen([python_exe, orchestrator_path], cwd=root_dir, env=env)
    return proc


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    start_time = time.time()
    orchestrator_proc = None

    try:
        while True:
            clear()
            print("1) Start integrated attack sprint")
            print("2) Stop running attack")
            print("3) Refresh metrics")
            print("4) Exit")
            print("")

            if orchestrator_proc and orchestrator_proc.poll() is None:
                print("[Orchestrator] Running — PID:", orchestrator_proc.pid)
            elif orchestrator_proc:
                print("[Orchestrator] Last run finished.")

            _print_metrics(start_time)

            choice = input("Select an option: ").strip()

            if choice == "1":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    input("Orchestrator already running. Press Enter to continue...")
                else:
                    print("\n--- Attack Configuration ---")
                    t_url = input("Target API URL (leave blank for default): ").strip()
                    a_key = input("Target API Key (leave blank for none):    ").strip()
                    
                    orchestrator_proc = run_orchestrator(root_dir, t_url or None, a_key or None)
                    input("Orchestrator started. Press Enter to continue...")

            elif choice == "2":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                    orchestrator_proc.wait(timeout=5)
                    input("Orchestrator terminated. Press Enter to continue...")
                else:
                    input("No orchestrator running. Press Enter to continue...")

            elif choice == "3":
                input("Refreshed. Press Enter to continue...")

            elif choice == "4":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                break

            else:
                input("Unknown option. Press Enter to continue...")

    except KeyboardInterrupt:
        if orchestrator_proc and orchestrator_proc.poll() is None:
            orchestrator_proc.terminate()
        print("\nDashboard stopped.")


if __name__ == "__main__":
    main()