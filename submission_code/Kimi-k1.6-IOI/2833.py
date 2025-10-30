from typing import List
from collections import defaultdict

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs based on their time
        logs.sort(key=lambda log: log[1])
        # Sort queries along with their original indices based on query time
        sorted_queries = sorted([(q, idx) for idx, q in enumerate(queries)], key=lambda x: x[0])
        # Initialize pointers and frequency map
        left = 0
        right = 0
        freq = defaultdict(int)
        unique_count = 0
        result = [0] * len(queries)
        
        # Process each query in sorted order
        for q, idx in sorted_queries:
            L = q - x
            R = q
            
            # Move right pointer to include all logs with time <= R
            while right < len(logs) and logs[right][1] <= R:
                server_id = logs[right][0]
                if freq[server_id] == 0:
                    unique_count += 1
                freq[server_id] += 1
                right += 1
            
            # Move left pointer to exclude logs with time < L
            while left < len(logs) and logs[left][1] < L:
                server_id = logs[left][0]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    unique_count -= 1
                left += 1
            
            # The number of inactive servers is total servers minus those active in the window
            result[idx] = n - unique_count
        
        return result