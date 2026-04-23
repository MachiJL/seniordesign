The objective of our Attack Team is to design, implement, and deploy an advanced attack framework capable of direct injection, RAG injection, and tool abuse attacks.


-The success_eval_logic.py, combined with Master_Orchestrator.py, forms a complete automated prompt injection attack framework.

- BasePrompts.txt

- LLM_Client_Adapter Adapter layer for communicating with real LLM APIs or the mock server.

- Master_Orchestrator.py acts as the delivery engine; it blasts the payloads at an LLM endpoint concurrently and rate-limited. 

- success_eval_logic.py serves as the scoring engine; it determines which payloads succeeded in compromising the LLM model.

- MOCK_API.py A lightweight local mock LLM server. Used for safe, fast, and repeatable testing without hitting real models.

- integrated_Orchestrator.py 
  
- payload_laoder.py
  
- rag_injection_framework.py
  
- rag_payloads.py
  
- rag_test_runner.py

- toolAbuse_payload

- tool_abuse_payloads.py

- tool_abuse_test_runner.py
  
