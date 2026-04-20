import asyncio
import os
import sys
from unicodedata import category
from tool_abuse_payloads import TOOL_ABUSE_METADATA, get_all_tool_payloads, get_tool_payloads_by_category

async def run_tool_category_test(orchestrator, category: str):
    metadata = TOOL_ABUSE_METADATA.get(category)
    if not metadata: return

    print(f"\n{'='*70}")
    print(f"TOOL ABUSE TEST: {category.upper().replace('_', ' ')}")
    print(f"Description: {metadata['description']}")
    print(f"Severity: {metadata['severity']}")
    print(f"{'='*70}\n")

    orchestrator.attack_category = f"Tool Abuse: {category}"
    # Temporarily disable mutation for targeted testing
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0
    orchestrator.mutation_cap = 0

    try:
        await orchestrator.run_attack_sprint(metadata['payloads'])
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

async def run_full_tool_suite(orchestrator):
    print(f"\n{'='*70}")
    print("FULL TOOL ABUSE VULNERABILITY ASSESSMENT")
    print(f"{'='*70}\n")
    
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0
    orchestrator.mutation_cap = 0
    
    try:
        await orchestrator.run_attack_sprint(get_all_tool_payloads())
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

async def run_tool_with_mutation(orchestrator):
    """Runs tool abuse tests with the mutation engine enabled."""
    print(f"\n{'='*70}")
    print("TOOL ABUSE WITH MUTATION ENGINE")
    print("="*70 + "\n")
    
    orig_expansion = orchestrator.expansion_factor
    orig_mutation = orchestrator.mutation_cap
    
    cap_input = input("Enter mutation cap (default 20): ").strip()
    orchestrator.mutation_cap = int(cap_input) if cap_input.isdigit() else 20
    # Ensure mutation is actually ON for this mode
    orchestrator.expansion_factor = 2 
    
    try:
        payloads = get_all_tool_payloads()
        await orchestrator.run_attack_sprint(payloads)
    finally:
        orchestrator.expansion_factor = orig_expansion
        orchestrator.mutation_cap = orig_mutation

def show_tool_menu():
    print("\n" + "="*70)
    print("TOOL ABUSE TEST SUITE - MENU")
    print("="*70)
    print("\n1. Full Tool Abuse Suite (All Categories)")
    print("2. Tool Discovery Only")
    print("3. Unauthorized Invocation Only")
    print("4. Parameter Manipulation Only")
    print("5. Output Exfiltration Only")
    print("6. Tool Chain Escalation Only")
    print("7. Tool Abuse with Mutation Engine")
    print("8. Exit")
    print("\n" + "="*70)
    return input("\nSelect option (1-8): ").strip()

async def tool_abuse_menu_handler(orchestrator):
    while True:
        choice = show_tool_menu()
        if choice == "1":
            await run_full_tool_suite(orchestrator)
        elif choice == "2":
            await run_tool_category_test(orchestrator, "discovery")
        elif choice == "3":
            await run_tool_category_test(orchestrator, "unauthorized_invocation")
        elif choice == "4":
            await run_tool_category_test(orchestrator, "parameter_manipulation")
        elif choice == "5":
            await run_tool_category_test(orchestrator, "output_exfiltration")
        elif choice == "6":
            await run_tool_category_test(orchestrator, "chain_escalation")
        elif choice == "7":
            await run_tool_with_mutation(orchestrator)
        elif choice == "8":
            break
        else:
            print("\nInvalid choice.")
        
        if choice in ["1", "2", "3", "4", "5", "6", "7"]:
            if input("\nRun another Tool Abuse test? (y/n): ").strip().lower() != 'y':
                break