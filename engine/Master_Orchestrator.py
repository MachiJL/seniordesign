# This is a simple implementation of the Master Controller using asyncio and aiohttp
# Made by Mose E. Willis Jr. for Senior Design 2026 
import asyncio
import aiohttp
import time

class CompoundMaster:
    def __init__(self, target_url, rate_limit=10):
        self.target_url = target_url
        self.rate_limit = rate_limit
        # We initialize these as None to prevent the "Different Loop" error
        self.queue = None
        self.session = None
        self.total_sent = 0

    async def slave_worker(self, worker_id):
        """Worker that pulls from the queue."""
        while True:
            payload = await self.queue.get()
            if payload is None:
                self.queue.task_done()
                break
            try:
                async with self.session.post(self.target_url, json={"prompt": payload}) as resp:
                    if resp.status == 200:
                        self.total_sent += 1
            except Exception:
                pass
            finally:
                self.queue.task_done()

    async def run_attack_sprint(self, payloads):
        # FIX: Initialize the Queue and Session INSIDE the active loop
        self.queue = asyncio.Queue()
        self.session = aiohttp.ClientSession()
        self.start_time = time.time()

        # Start 5 workers
        workers = [asyncio.create_task(self.slave_worker(i)) for i in range(5)]

        # Feed payloads
        for i, p in enumerate(payloads):
            await self.queue.put(p)
            await asyncio.sleep(1 / self.rate_limit)

        await self.queue.join()
        for _ in range(5): await self.queue.put(None)
        await asyncio.gather(*workers)
        await self.session.close()
        
        print(f"Finished! Total sent: {self.total_sent}")

if __name__ == "__main__":
    mock_data = [f"Test {i}" for i in range(20)]
    # Using a public test endpoint to ensure it works immediately!
    master = CompoundMaster(target_url="https://httpbin.org/post")
    
    print("Starting Master Controller...")
    asyncio.run(master.run_attack_sprint(mock_data))
