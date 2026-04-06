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

    print("═" * 70)
    print("      RED TEAM ATTACK FRAMEWORK – LIVE DASHBOARD")
    print("═" * 70)

    if not metrics:
        print("Status: Waiting for metrics data...")
    else:
        total = metrics.get("total_sent", 0)
        success = metrics.get("success", 0)
        errors = metrics.get("errors", 0)
        pps = metrics.get("pps", 0)
        avg_latency = metrics.get("avg_latency_ms", 0)

        success_rate = (success / total * 100) if total > 0 else 0

        print(f"Status       : RUNNING")
        print(f"Runtime      : {runtime:.1f} seconds")
        print("-" * 60)
        print(f"Requests/sec (PPS)     : {pps}")
        print(f"Total Requests Sent    : {total}")
        print(f"Successful Bypasses    : {success}")
        print(f"Errors                 : {errors}")
        print(f"Success Rate           : {success_rate:.2f}%")
        print(f"Average Latency        : {avg_latency:.2f} ms")
        print("-" * 60)

        if "last_event" in metrics and metrics["last_event"]:
            print(f"Last Event   : {metrics['last_event'][:140]}...")

    print("═" * 70)


def run_orchestrator(root_dir):
    python_exe = sys.executable
    orchestrator_path = os.path.join(root_dir, "intergrated_orchestrator.py")
    
    print(f"\n[DEBUG] Root directory     : {root_dir}")
    print(f"[DEBUG] Looking for file    : {orchestrator_path}")
    
    if not os.path.exists(orchestrator_path):
        print(f"[ERROR] ❌ Could not find intergrated_orchestrator.py")
        print(f"        Expected path: {orchestrator_path}")
        print("        Make sure the orchestrator file is in the main folder (same level as CLI folder)")
        input("\nPress Enter to continue...")
        return None
    
    print(f"[SUCCESS] Orchestrator file found. Launching...")

    env = os.environ.copy()
    env["LAUNCH_DASHBOARD"] = "0"   # Prevent recursive spawning

    try:
        proc = subprocess.Popen(
            [python_exe, orchestrator_path],
            cwd=root_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print(f"[SUCCESS] ✅ Orchestrator started! PID: {proc.pid}")
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to start orchestrator: {e}")
        input("\nPress Enter to continue...")
        return None


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    start_time = time.time()
    orchestrator_proc = None

    print(f"[INFO] Dashboard started. Root folder: {root_dir}\n")

    try:
        while True:
            clear()
            print("1) Start integrated attack sprint")
            print("2) Stop running attack")
            print("3) Refresh metrics")
            print("4) Exit")
            print("")

            if orchestrator_proc and orchestrator_proc.poll() is None:
                print(f"[Orchestrator] ✅ Running — PID: {orchestrator_proc.pid}")
            elif orchestrator_proc:
                print("[Orchestrator] Stopped or finished.")

            _print_metrics(start_time)

            choice = input("Select an option: ").strip()

            if choice == "1":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    input("Orchestrator is already running. Press Enter...")
                else:
                    orchestrator_proc = run_orchestrator(root_dir)
                    input("\nPress Enter to continue...")

            elif choice == "2":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                    try:
                        orchestrator_proc.wait(timeout=5)
                    except:
                        pass
                    print("Orchestrator terminated.")
                    input("Press Enter to continue...")
                else:
                    input("No orchestrator is running. Press Enter...")

            elif choice == "3":
                print("Metrics refreshed.")
                input("Press Enter to continue...")

            elif choice == "4":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                print("\nDashboard closed.")
                break

            else:
                input("Unknown option. Press Enter to continue...")

    except KeyboardInterrupt:
        if orchestrator_proc and orchestrator_proc.poll() is None:
            orchestrator_proc.terminate()
        print("\nDashboard stopped by user.")


if __name__ == "__main__":
    main()