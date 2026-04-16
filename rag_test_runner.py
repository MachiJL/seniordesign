"""
RAG Injection Test Runner
Complete example of integrating RAG injection tests with integrated_orchestrator.py
"""

import asyncio
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intergrated_orchestrator import CompoundMaster
from rag_injection_framework import (
    run_rag_injection_tests_with_orchestrator,
    create_rag_payload_list,
    RAGInjectionAdapter,
    get_attack_vectors
)
from rag_payloads import ATTACK_VECTOR_METADATA


async def run_full_rag_test_suite(orchestrator: CompoundMaster):
    """
    Run comprehensive RAG injection tests across all attack vectors
    """
    # Configuration
    # Use orchestrator's configuration
    SYSTEM_PROMPT = orchestrator.system_instruction
    MOCK_URL = orchestrator.mock_base_url
    API_KEY = orchestrator.api_key
    
    print("\n" + "="*70)
    print("RAG INJECTION VULNERABILITY ASSESSMENT")
    print("="*70)
    print(f"\nTarget: {MOCK_URL}")
    print(f"System Prompt: {SYSTEM_PROMPT[:60]}...\n")
    
    # Display attack vectors being tested
    print("Testing Attack Vectors:")
    for vector in get_attack_vectors():
        metadata = ATTACK_VECTOR_METADATA[vector]
        print(f"\n  [{vector.upper()}]")
        print(f"  Description: {metadata['description']}")
        print(f"  Severity: {metadata['severity']}")
        print(f"  Detection Difficulty: {metadata['detection_difficulty']}")
        print(f"  Payloads: {len(metadata['payload_ids'])}")
    
    print("\n" + "="*70)
    print()
    
    # Temporarily ensure orchestrator's mutation settings are off for this specific test
    original_expansion_factor = orchestrator.expansion_factor
    original_mutation_cap = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0
    
    # Run RAG injection tests
    await run_rag_injection_tests_with_orchestrator(
        orchestrator=orchestrator,
        attack_vectors=None,  # Test all vectors
        docs_per_vector=2  # 2 documents per vector for comprehensive coverage
    ) 

    # Restore original orchestrator settings
    orchestrator.expansion_factor = original_expansion_factor
    orchestrator.mutation_cap = original_mutation_cap


async def run_targeted_rag_test(orchestrator: CompoundMaster, attack_vector: str):
    """
    Run tests for a specific RAG attack vector
    
    Args:
        attack_vector: One of: document_injection, retrieval_manipulation,
        embedding_poisoning, context_overflow, indirect_injection
    """
    SYSTEM_PROMPT = orchestrator.system_instruction
    MOCK_URL = orchestrator.mock_base_url
    API_KEY = orchestrator.api_key
    
    print(f"\n{'='*70}")
    print(f"TARGETED RAG TEST: {attack_vector.upper()}")
    print(f"{'='*70}\n")
    
    metadata = ATTACK_VECTOR_METADATA[attack_vector]
    print(f"Description: {metadata['description']}")
    print(f"Severity: {metadata['severity']}")
    print(f"Detection Difficulty: {metadata['detection_difficulty']}\n")
    
    original_expansion_factor = orchestrator.expansion_factor
    original_mutation_cap = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0

    await run_rag_injection_tests_with_orchestrator(
        orchestrator=orchestrator,
        attack_vectors=[attack_vector],  # Test only this vector
        docs_per_vector=3  # More thorough testing of single vector
    )


async def run_simple_rag_test(orchestrator: CompoundMaster):
    """
    Simplified RAG test using direct payloads without document generation
    Good for quick testing
    """
    SYSTEM_PROMPT = orchestrator.system_instruction
    MOCK_URL = orchestrator.mock_base_url
    API_KEY = orchestrator.api_key
    
    print("\n" + "="*70)
    print("SIMPLE RAG PAYLOAD TEST")
    print("="*70 + "\n")
    
    # Create simple payload list
    payloads = create_rag_payload_list(
        attack_vectors=['document_injection', 'embedding_poisoning'],
        max_per_vector=3
    )
    
    print(f"Testing {len(payloads)} RAG injection payloads\n")
    original_expansion_factor = orchestrator.expansion_factor
    original_mutation_cap = orchestrator.mutation_cap
    orchestrator.expansion_factor = 0  # Disable mutation for simple test
    orchestrator.mutation_cap = 0      # No mutations for simple test
    
    try:
        await orchestrator.run_attack_sprint(payloads)
    finally:
        orchestrator.expansion_factor = original_expansion_factor
        orchestrator.mutation_cap = original_mutation_cap


async def run_rag_with_mutation(orchestrator: CompoundMaster):
    """
    Run RAG tests WITH the mutator enabled
    This will evolve successful RAG injections
    """
    SYSTEM_PROMPT = orchestrator.system_instruction
    MOCK_URL = orchestrator.mock_base_url
    API_KEY = orchestrator.api_key
    
    print("\n" + "="*70)
    print("RAG INJECTION WITH MUTATION ENGINE")
    print("="*70 + "\n")
    print("This test will:")
    print("1. Test RAG injection payloads")
    print("2. Use LLM mutator to evolve successful attacks")
    print("3. Generate variant payloads for discovered vulnerabilities\n")
    
    # Temporarily override orchestrator's expansion_factor and mutation_cap for this specific test
    original_expansion_factor = orchestrator.expansion_factor
    original_mutation_cap = orchestrator.mutation_cap
    orchestrator.expansion_factor = 2  # Enable mutation expansion
    orchestrator.mutation_cap = 20  # Allow up to 20 mutations
    try:
        await run_rag_injection_tests_with_orchestrator(
            orchestrator=orchestrator,
            attack_vectors=['document_injection', 'indirect_injection'],
            docs_per_vector=2
        )
    finally:
        # Restore original orchestrator settings
        orchestrator.expansion_factor = original_expansion_factor
        orchestrator.mutation_cap = original_mutation_cap

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*70)
    print("RAG INJECTION TEST SUITE - MENU")
    print("="*70)
    print("\n1. Full RAG Test Suite (All Attack Vectors)")
    print("2. Document Injection Only")
    print("3. Retrieval Manipulation Only")
    print("4. Embedding Poisoning Only")
    print("5. Context Window Overflow Only")
    print("6. Indirect Injection Only")
    print("7. Simple RAG Test (Quick)")
    print("8. RAG Test with Mutation Engine")
    print("9. Exit")
    print("\n" + "="*70)
    
    choice = input("\nSelect option (1-9): ").strip()
    return choice


async def rag_menu_handler(orchestrator: CompoundMaster):
    """
    Handles the interactive RAG menu and dispatches to the appropriate test function.
    This function will be called by intergrated_orchestrator.py
    """
    
    while True:
        choice = show_menu()
        
        if choice == "1":
            await run_full_rag_test_suite(orchestrator)
        elif choice == "2":
            await run_targeted_rag_test(orchestrator, "document_injection")
        elif choice == "3":
            await run_targeted_rag_test(orchestrator, "retrieval_manipulation")
        elif choice == "4":
            await run_targeted_rag_test(orchestrator, "embedding_poisoning")
        elif choice == "5":
            await run_targeted_rag_test(orchestrator, "context_overflow")
        elif choice == "6":
            await run_targeted_rag_test(orchestrator, "indirect_injection")
        elif choice == "7":
            await run_simple_rag_test(orchestrator)
        elif choice == "8":
            await run_rag_with_mutation(orchestrator)
        elif choice == "9":
            print("\nExiting RAG test suite menu.")
            break
        else:
            print("\nInvalid choice. Please select 1-9.")
        
        # Ask if user wants to continue
        if choice in ["1", "2", "3", "4", "5", "6", "7", "8"]:
            cont = input("\nRun another RAG test from this menu? (y/n): ").strip().lower()
            if cont != 'y':
                print("\nReturning to main dashboard menu.")
                break


if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║          RAG INJECTION VULNERABILITY TESTING FRAMEWORK           ║
║                                                                  ║
║  Tests 5 Attack Vectors:                                        ║
║  1. Document Injection        - Direct malicious instructions   ║
║  2. Retrieval Manipulation    - Priority/ranking exploitation   ║
║  3. Embedding Poisoning       - Semantic vector poisoning       ║
║  4. Context Window Overflow   - Context limit exploitation      ║
║  5. Indirect Injection        - Conditional trigger attacks     ║
║                                                                  ║
║  Integrates with: integrated_orchestrator.py                    ║
╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check if running in automated mode
    if len(sys.argv) > 1:
        # This block is for standalone testing of rag_test_runner.py
        # If it's called from orchestrator, rag_menu_handler will be invoked directly.
        # For standalone, we need a dummy orchestrator.
        print("Running RAG Test Runner in standalone mode (using default orchestrator config).")
        dummy_orchestrator = CompoundMaster(
            rate_limit=5,
            system_instruction="You are a helpful assistant with knowledge base access.",
            mock_base_url=os.getenv("TARGET_API_URL", "http://127.0.0.1:8001"),
            api_key=os.getenv("TARGET_API_KEY", "cyborgs-local-client-key"),
            expansion_factor=0, # Default to no expansion for standalone RAG menu
            mutation_cap=0,     # Default to no mutation for standalone RAG menu
            launch_dashboard=False # Never launch dashboard from here
        )
        asyncio.run(rag_menu_handler(dummy_orchestrator))
    else:
        # Interactive mode
        print("Running RAG Test Runner in standalone mode (using default orchestrator config).")
        dummy_orchestrator = CompoundMaster(
            rate_limit=5,
            system_instruction="You are a helpful assistant with knowledge base access.",
            mock_base_url=os.getenv("TARGET_API_URL", "http://127.0.0.1:8001"),
            api_key=os.getenv("TARGET_API_KEY", "cyborgs-local-client-key"),
            expansion_factor=0, # Default to no expansion for standalone RAG menu
            mutation_cap=0,     # Default to no mutation for standalone RAG menu
            launch_dashboard=False # Never launch dashboard from here
        )
        asyncio.run(rag_menu_handler(dummy_orchestrator))
