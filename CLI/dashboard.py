import time
import json
import os
import subprocess
import sys
<<<<<<< HEAD
=======
from urllib import request, error
>>>>>>> origin/main

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")


<<<<<<< HEAD
=======
def get_mock_api_status():
    target = os.getenv("TARGET_API_URL", "http://127.0.0.1:8001")
    base = target.rstrip("/")
    if base.endswith("/chat"):
        base = base[:-5]
    health_url = f"{base}/health"
    try:
        with request.urlopen(health_url, timeout=1.5) as resp:
            if resp.status == 200:
                return f"ONLINE ({health_url})"
    except error.URLError:
        pass
    except Exception:
        pass
    return f"OFFLINE ({health_url})"


>>>>>>> origin/main
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


<<<<<<< HEAD
=======
def continuous_refresh():
    print("Entering continuous refresh mode. Press Ctrl+C to stop.")
    try:
        while True:
            clear()
            _print_metrics(time.time())  # Use current time for runtime
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nExiting continuous refresh mode.")


>>>>>>> origin/main
def _print_metrics(start_time):
    metrics = load_metrics()
    runtime = time.time() - start_time

<<<<<<< HEAD
    print("═" * 60)
    print("      RED TEAM ATTACK FRAMEWORK – LIVE DASHBOARD")
    print("═" * 60)

    if not metrics:
        print("Status: Waiting for metrics data...")
    else:
=======
    print("═" * 70)
    print("      RED TEAM ATTACK FRAMEWORK – LIVE DASHBOARD")
    print("═" * 70)
    print(f"Mock API Status: {get_mock_api_status()}")

    if not metrics:
        print("Status: Waiting for metrics data...")
    elif "final_summary" in metrics:
        # Display final summary
        fs = metrics["final_summary"]
        print("Status: COMPLETED")
        print("-" * 60)
        print(f"Target Model           : {fs.get('target_model', 'Unknown')}")
        print(f"Total tests run        : {fs.get('total_tests', 0)} (inc. auto-mutations)")
        print(f"Successful Bypasses    : {fs.get('successful_bypasses', 0)}")
        print(f"Success rate           : {fs.get('success_rate', 0):.1%}")
        print(f"Average confidence     : {fs.get('average_confidence', 0):.2f}")
        print(f"Avg API Latency        : {fs.get('avg_api_latency', 0):.2f}s")
        print(f"Total duration         : {fs.get('total_duration', 0):.2f}s")
        print(f"Worker Health          : {fs.get('worker_health', 'Unknown')}")
        attack_dist = fs.get("attack_type_distribution", {})
        if attack_dist:
            print("Attack type distribution:")
            for attack_type, count in attack_dist.items():
                print(f"  - {attack_type:<28} {count}")
        severity_dist = fs.get("severity_distribution", {})
        if severity_dist:
            print("Severity distribution :")
            for severity, count in severity_dist.items():
                print(f"  - {severity:<28} {count}")
    else:
        # Live metrics
>>>>>>> origin/main
        total = metrics.get("total_sent", 0)
        success = metrics.get("success", 0)
        errors = metrics.get("errors", 0)
        pps = metrics.get("pps", 0)
        avg_latency = metrics.get("avg_latency_ms", 0)

        success_rate = (success / total * 100) if total > 0 else 0

<<<<<<< HEAD
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
            print(f"Last Event: {metrics['last_event'][:150]}...")

    print("═" * 60)


def run_orchestrator(root_dir):
    python_exe = sys.executable
    orchestrator_path = os.path.join(root_dir, "intergrated_orchestrator.py")
    
    print(f"\n[DEBUG] Root directory: {root_dir}")
    print(f"[DEBUG] Looking for orchestrator at: {orchestrator_path}")
    
    if not os.path.exists(orchestrator_path):
        print(f"[ERROR] ❌ intergrated_orchestrator.py NOT FOUND!")
        print(f"        Expected location: {orchestrator_path}")
        print("        Make sure the file is in the main folder (one level above CLI)")
=======
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


def run_orchestrator(root_dir, attack_mode="combined"):
    python_exe = sys.executable
    orchestrator_path = os.path.join(root_dir, "intergrated_orchestrator.py")
    
    print(f"\n[DEBUG] Root directory     : {root_dir}")
    print(f"[DEBUG] Looking for file    : {orchestrator_path}")
    
    if not os.path.exists(orchestrator_path):
        print(f"[ERROR] ❌ Could not find intergrated_orchestrator.py")
        print(f"        Expected path: {orchestrator_path}")
        print("        Make sure the orchestrator file is in the main folder (same level as CLI folder)")
>>>>>>> origin/main
        input("\nPress Enter to continue...")
        return None
    
    print(f"[SUCCESS] Orchestrator file found. Launching...")

    env = os.environ.copy()
<<<<<<< HEAD
    env["LAUNCH_DASHBOARD"] = "0"   # Prevent recursive dashboard spawning
=======
    env["LAUNCH_DASHBOARD"] = "0"   # Prevent recursive spawning
    env["ATTACK_MODE"] = attack_mode
>>>>>>> origin/main

    try:
        proc = subprocess.Popen(
            [python_exe, orchestrator_path],
            cwd=root_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
<<<<<<< HEAD
        print(f"[SUCCESS] ✅ Orchestrator started successfully! PID: {proc.pid}")
=======
        print(f"[SUCCESS] ✅ Orchestrator started! PID: {proc.pid}")
>>>>>>> origin/main
        return proc
    except Exception as e:
        print(f"[ERROR] Failed to start orchestrator: {e}")
        input("\nPress Enter to continue...")
        return None


def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    start_time = time.time()
    orchestrator_proc = None

<<<<<<< HEAD
    print(f"[INFO] Dashboard started. Root folder detected as: {root_dir}")
=======
    print(f"[INFO] Dashboard started. Root folder: {root_dir}\n")
>>>>>>> origin/main

    try:
        while True:
            clear()
<<<<<<< HEAD
            print("1) Start integrated attack sprint")
            print("2) Stop running attack")
            print("3) Refresh metrics")
            print("4) Exit")
=======
            print("1) Start automated script attack")
            print("2) Start tool abuse attack")
            print("3) Start combined attack sprint")
            print("4) Stop running attack")
            print("5) Reset metrics")
            print("6) Refresh display")
            print("7) Continuous refresh mode")
            print("8) Exit")
>>>>>>> origin/main
            print("")

            if orchestrator_proc and orchestrator_proc.poll() is None:
                print(f"[Orchestrator] ✅ Running — PID: {orchestrator_proc.pid}")
            elif orchestrator_proc:
<<<<<<< HEAD
                print("[Orchestrator] Finished or stopped.")
=======
                print("[Orchestrator] Stopped or finished.")
>>>>>>> origin/main

            _print_metrics(start_time)

            choice = input("Select an option: ").strip()

<<<<<<< HEAD
            if choice == "1":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    input("Orchestrator is already running. Press Enter...")
                else:
                    orchestrator_proc = run_orchestrator(root_dir)
                    input("\nPress Enter to continue...")

            elif choice == "2":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                    orchestrator_proc.wait(timeout=5)
=======
            if choice in {"1", "2", "3"}:
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    input("Orchestrator is already running. Press Enter...")
                else:
                    mode_map = {
                        "1": "script",
                        "2": "tool",
                        "3": "combined",
                    }
                    selected_mode = mode_map[choice]
                    orchestrator_proc = run_orchestrator(root_dir, selected_mode)
                    input(f"Orchestrator started in '{selected_mode}' mode. Press Enter to continue...")

            elif choice == "4":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                    try:
                        orchestrator_proc.wait(timeout=5)
                    except:
                        pass
>>>>>>> origin/main
                    print("Orchestrator terminated.")
                    input("Press Enter to continue...")
                else:
                    input("No orchestrator is running. Press Enter...")

<<<<<<< HEAD
            elif choice == "3":
                print("Metrics refreshed.")
                input("Press Enter to continue...")

            elif choice == "4":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                break

            else:
                input("Unknown option. Press Enter to continue...")
=======
            elif choice == "5":
                if os.path.exists(METRICS_FILE):
                    os.remove(METRICS_FILE)
                input("Metrics reset. Press Enter to return...")

            elif choice == "6":
                input("Display refreshed. Press Enter to return...")

            elif choice == "7":
                continuous_refresh()

            elif choice == "8":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                print("\nDashboard closed.")
                break

            else:
                input("Unknown option. Press Enter to return...")
>>>>>>> origin/main

    except KeyboardInterrupt:
        if orchestrator_proc and orchestrator_proc.poll() is None:
            orchestrator_proc.terminate()
<<<<<<< HEAD
        print("\nDashboard stopped.")
=======
        print("\nDashboard stopped by user.")
>>>>>>> origin/main


if __name__ == "__main__":
    main()