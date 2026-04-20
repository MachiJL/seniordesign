import asyncio
from script_payloads import SCRIPT_ATTACK_METADATA, get_all_script_payloads

async def run_script_category_test(orchestrator, category: str):
    metadata = SCRIPT_ATTACK_METADATA.get(category)
    if not metadata: return

    print(f"\n{'='*70}")
    print(f"PROMPT INJECTION TEST: {category.upper().replace('_', ' ')}")
    print(f"Description: {metadata['description']}")
    print(f"Severity: {metadata['severity']}")
    print(f"{'='*70}\n")

    orchestrator.attack_category = f"Script Injection: {category}"
    # Targeted testing usually disables expansion to see base payload performance
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0
    orchestrator.mutation_cap = 0

    try:
        await orchestrator.run_attack_sprint(metadata['payloads'])
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

async def run_full_script_suite(orchestrator):
    print(f"\n{'='*70}")
    print("FULL PROMPT INJECTION VULNERABILITY ASSESSMENT")
    print(f"{'='*70}\n")
    
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0
    orchestrator.mutation_cap = 0
    
    try:
        await orchestrator.run_attack_sprint(get_all_script_payloads())
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

async def run_script_with_mutation(orchestrator):
    """Runs prompt injection with high mutation settings."""
    print(f"\n{'='*70}")
    print("MUTATION-DRIVEN PROMPT INJECTION")
    print("="*70 + "\n")
    
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    
    cap_input = input("Enter mutation cap (default 20): ").strip()
    orchestrator.mutation_cap = int(cap_input) if cap_input.isdigit() else 20
    orchestrator.expansion_factor = 3
    
    try:
        payloads = get_all_script_payloads()
        await orchestrator.run_attack_sprint(payloads)
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

def show_script_menu():
    print("\n" + "="*70)
    print("PROMPT INJECTION TEST SUITE - MENU")
    print("="*70)
    print("\n1. Full Script Suite (All 50 Payloads)")
    print("2. Direct Instruction Override Only")
    print("3. Role/Context Manipulation Only")
    print("4. Delimiter/Encoding Confusion Only")
    print("5. Nested/Indirect Injection Only")
    print("6. Prompt Leakage/Meta Exploitation Only")
    print("7. Script Attack with Mutation Engine")
    print("8. Exit")
    print("\n" + "="*70)
    return input("\nSelect option (1-8): ").strip()

async def script_menu_handler(orchestrator):
    while True:
        choice = show_script_menu()
        if choice == "1":
            await run_full_script_suite(orchestrator)
        elif choice == "2":
            await run_script_category_test(orchestrator, "direct_override")
        elif choice == "3":
            await run_script_category_test(orchestrator, "role_manipulation")
        elif choice == "4":
            await run_script_category_test(orchestrator, "delimiter_confusion")
        elif choice == "5":
            await run_script_category_test(orchestrator, "indirect_injection")
        elif choice == "6":
            await run_script_category_test(orchestrator, "meta_exploitation")
        elif choice == "7":
            await run_script_with_mutation(orchestrator)
        elif choice == "8":
            break
        else:
            print("\nInvalid choice.")
        
        if choice in ["1", "2", "3", "4", "5", "6", "7"]:
            if input("\nRun another Script Injection test? (y/n): ").strip().lower() != 'y':
                break