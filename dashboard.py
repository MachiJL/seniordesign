import time

def main():
    print("=== Red Team Attack Framework Dashboard ===")
    print("Status: Initializing...\n")

    pps = 0
    success_rate = 0.0

    try:
        while True:
            pps += 1
            success_rate = min(success_rate + 0.2, 100)

            print(f"PPS: {pps}")
            print(f"Success Rate: {success_rate:.1f}%")
            print("-" * 40)

            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDashboard stopped.")

if __name__ == "__main__":
    main()
