The objective of our Attack Team is to design, implement, and deploy an advanced attack framework capable of direct injection, RAG injection, and tool abuse attacks.


The success_eval_logic.py, combined with Master_Orchestrator.py, forms a complete automated prompt injection attack framework.

- BasePrompts.txt

- LLM_Client_Adapter

- Master_Orchestrator.py acts as the delivery engine; it blasts the payloads at an LLM endpoint concurrently and rate-limited. 

- success_eval_logic.py serves as the scoring engine; it determines which payloads succeeded in compromising the LLM model.
