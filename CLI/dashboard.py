import time
import json
import os
import subprocess
import sys
import threading
from collections import deque
from urllib import request, error

METRICS_FILE = os.path.join(os.path.dirname(__file__), "metrics.json")
LOG_BUFFER = deque(maxlen=20)  # Keep the last 20 lines of orchestrator output for better visibility

BANNER = r"""
   ___             _      ___                _            
  / _ \___ ___ _ (_)___ / _ )_______ ___ _ / /_____ ____ 
 / __ / -_) _ `/ / (_-< / _  / __/ -_) _ `//  '_/ -_) __/ 
/_/ |_\___\_, / /_//___/____/_/  \__/\_,_//_/\_\\__/_/    
         /___/                                            
                  ___
                 |   |
              _  |   |  _
             | |_|   |_| |
             |           |
             |___|###|___|
              \_       _/
                | [ ] |
                |     |   / \ / \
                |     |  (   X   )
                |     |   \ / \ /
                |     |    /   \
                |     |
                \_____/
"""


def get_mock_api_status(target_url):
    base = target_url.rstrip("/")
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


def log_reader(proc):
    """Thread function to read lines from orchestrator stdout."""
    try:
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ''):
            if line:
                LOG_BUFFER.append(line.strip())
    except Exception:
        pass
    finally:
        proc.stdout.close()


def continuous_refresh(start_time, target_url, proc=None):
    if proc:
        # Clear buffer for new run
        LOG_BUFFER.clear()
        t = threading.Thread(target=log_reader, args=(proc,), daemon=True)
        t.start()

    print("Entering continuous refresh mode. Press Ctrl+C to stop.")
    try:
        while True:
            clear()
            print(BANNER)
            _print_metrics(start_time, target_url, list(LOG_BUFFER))
            if proc and proc.poll() is not None:
                print("\n[INFO] Orchestrator has finished execution.")
                input("Press Enter to return to menu...")
                break
            time.sleep(0.5) # Faster refresh for real-time feel
    except KeyboardInterrupt:
        print("\nExiting continuous refresh mode.")


def _print_metrics(start_time, target_url, logs=None):
    metrics = load_metrics()
    runtime = time.time() - start_time

    print("=" * 70)
    print("             AEGIS BREAKER - LIVE DASHBOARD")
    print("=" * 70)
    print(f"Mock API Status: {get_mock_api_status(target_url)}")

    if not metrics:
        print("Status: Waiting for metrics data...")
    else:
        is_final = "final_summary" in metrics
        data = metrics["final_summary"] if is_final else metrics
        
        status = "COMPLETED" if is_final else "RUNNING"
        total = data.get("total_tests") if is_final else data.get("total_sent", 0)
        success = data.get("successful_bypasses") if is_final else data.get("success", 0)
        category = data.get("category", "General")
        # Convert seconds to ms if final, else use stored ms
        latency = (data.get("avg_api_latency", 0) * 1000) if is_final else data.get("avg_latency_ms", 0)
        success_rate = (data.get("success_rate", 0) * 100) if is_final else ((success / total * 100) if total > 0 else 0)

        print(f"Status       : {status}")
        print(f"Attack Type  : {category}")
        print(f"Target Model : {data.get('target_model', 'mock-vulnerable-llm-v2')}")
        print(f"Runtime      : {data.get('total_duration', runtime) if is_final else runtime:.1f}s")
        print("-" * 60)
        
        print(f"Total Requests         : {total}")
        print(f"Successful Bypasses    : {success}")
        print(f"Success Rate           : {success_rate:.2f}%")
        print(f"Avg API Latency        : {latency:.2f} ms")
        
        if is_final:
            print(f"Worker Health          : {data.get('worker_health', 'Unknown')}")
            attack_dist = data.get("attack_type_distribution", {})
            if attack_dist:
                print("\nDetection Analytics (Success Indicators Detected):")
                for attack_type, count in attack_dist.items():
                    print(f"  - {attack_type:<25} {count}")
            severity_dist = data.get("severity_distribution", {})
            if severity_dist:
                print("\nRisk Assessment (Severity Distribution):")
                for severity in ['critical', 'high', 'medium', 'low']:
                    count = severity_dist.get(severity, 0)
                    print(f"  - {severity.upper():<25} {count}")
        else:
            print(f"Requests/sec (PPS)     : {data.get('pps', 0)}")
            if data.get("last_event"):
                print("-" * 60)
                print(f"Last Event   : {data['last_event'][:140]}...")

    if logs:
        print("-" * 60)
        print("RECENT ATTACK LOGS:")
        for log in logs:
            print(f"  > {log}")

    print("=" * 70)


def run_orchestrator(root_dir, attack_mode="combined", target_url=None, target_api_key=None, interactive=False):
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
    env["ATTACK_MODE"] = attack_mode
    if target_url: env["TARGET_API_URL"] = target_url
    if target_api_key: env["TARGET_API_KEY"] = target_api_key
    env["PYTHONUNBUFFERED"] = "1"   # Force real-time log flushing

    try:
        if interactive:
            # In interactive mode, we don't pipe so the child process can use the terminal
            stdout, stderr = None, None
        else:
            stdout, stderr = subprocess.PIPE, subprocess.STDOUT

        proc = subprocess.Popen(
            [python_exe, orchestrator_path],
            cwd=root_dir,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
            bufsize=1
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

    # Load initial target configuration
    target_url = os.getenv("TARGET_API_URL", "http://127.0.0.1:8001")
    target_api_key = os.getenv("TARGET_API_KEY", "cyborgs-local-client-key")

    print(f"[INFO] Aegis Breaker Dashboard started. Root folder: {root_dir}\n")

    try:
        while True:
            clear()
            print(BANNER)
            if orchestrator_proc and orchestrator_proc.poll() is None:
                print(f"[Orchestrator] ✅ Running — PID: {orchestrator_proc.pid}")
            elif orchestrator_proc:
                print("[Orchestrator] Stopped or finished.")

            _print_metrics(start_time, target_url)

            print("1) Start automated script attack")
            print("2) Start tool abuse attack")
            print("3) Start RAG injection attack")
            print("4) Start combined attack sprint")
            print("5) Stop running attack")
            print("6) Reset metrics")
            print("7) Refresh display")
            print("8) Continuous refresh mode")
            print("9) Configure Target API (URL/Key)")
            print("10) Exit")
            print("")

            choice = input("Select an option: ").strip()

            if choice in {"1", "2", "3", "4"}:
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    input("Orchestrator is already running. Press Enter...")
                    continue

                mode_map = {"1": "script", "2": "tool", "3": "rag", "4": "combined"}
                selected_mode = mode_map[choice]
                # All current modes are interactive, requiring terminal input for menus or configuration
                orchestrator_proc = run_orchestrator(root_dir, selected_mode, target_url, target_api_key, interactive=True)
                if orchestrator_proc:
                    start_time = time.time()
                    print(f"[INFO] {selected_mode.upper()} Mode active. Interaction required in orchestrator console.")
                    orchestrator_proc.wait()
                    input("\nAttack finished. Press Enter to return to menu...")

            elif choice == "5":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                    orchestrator_proc.wait()
                    print("[INFO] Attack sprint stopped mid-way.")
                    input("Press Enter to continue...")
                else:
                    input("No orchestrator is running. Press Enter...")

            elif choice == "6":
                try: os.remove(METRICS_FILE)
                except OSError: pass
                start_time = time.time()
                LOG_BUFFER.clear()
                input("Metrics reset. Press Enter to return...")

            elif choice == "7":
                pass

            elif choice == "8":
                continuous_refresh(start_time, target_url)

            elif choice == "9":
                print(f"\n[CONFIG] Current URL: {target_url}")
                u = input("Enter new Target URL (leave blank to keep): ").strip()
                if u: target_url = u
                print(f"[CONFIG] Current Key: {target_api_key}")
                k = input("Enter new API Key (leave blank to keep): ").strip()
                if k: target_api_key = k
                input("\nSettings updated. Press Enter to return...")

            elif choice == "10":
                if orchestrator_proc and orchestrator_proc.poll() is None:
                    orchestrator_proc.terminate()
                print("\nDashboard closed.")
                break

            else:
                input("Unknown option. Press Enter to return...")

    except KeyboardInterrupt:
        if orchestrator_proc and orchestrator_proc.poll() is None:
            orchestrator_proc.terminate()
        print("\nDashboard stopped by user.")


if __name__ == "__main__":
    main()