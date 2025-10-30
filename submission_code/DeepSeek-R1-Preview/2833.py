import bisect
from collections import defaultdict
from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        times = [log[1] for log in logs]
        m = len(logs)
        
        # Prepare the queries with their L, R, and original index
        processed_queries = []
        for idx, q in enumerate(queries):
            s = q - x
            e = q
            L = bisect.bisect_left(times, s)
            R = bisect.bisect_right(times, e) - 1
            processed_queries.append((L, R, idx))
        
        # Determine block size for Mo's algorithm
        block_size = int(m ** 0.5) if m else 1
        
        # Define custom sort for Mo's algorithm
        def mo_cmp(query):
            L, R, idx = query
            block = L // block_size
            return (block, R if block % 2 == 0 else -R)
        
        # Sort queries according to Mo's order
        processed_queries.sort(key=mo_cmp)
        
        # Initialize variables for Mo's algorithm
        current_L = 0
        current_R = -1
        freq = defaultdict(int)
        unique_count = 0
        answers = [0] * len(queries)
        
        for q in processed_queries:
            L, R, idx = q
            # Extend current_R to R
            while current_R < R:
                current_R += 1
                server_id = logs[current_R][0]
                freq[server_id] += 1
                if freq[server_id] == 1:
                    unique_count += 1
            # Reduce current_R to R
            while current_R > R:
                server_id = logs[current_R][0]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    unique_count -= 1
                current_R -= 1
            # Reduce current_L to L
            while current_L < L:
                server_id = logs[current_L][0]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    unique_count -= 1
                current_L += 1
            # Extend current_L to L
            while current_L > L:
                current_L -= 1
                server_id = logs[current_L][0]
                freq[server_id] += 1
                if freq[server_id] == 1:
                    unique_count += 1
            # Calculate the answer
            answers[idx] = n - unique_count
        
        return answers