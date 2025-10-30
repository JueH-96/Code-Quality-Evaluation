from typing import List
from collections import defaultdict

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time (second element of each log)
        sorted_logs = sorted(logs, key=lambda log: log[1])
        # Prepare queries with their original indices and sort by query time
        sorted_queries = sorted([(q, i) for i, q in enumerate(queries)], key=lambda x: x[0])
        
        left = 0
        right = 0
        server_counts = defaultdict(int)
        unique = 0
        ans = [0] * len(queries)
        
        for q, original_idx in sorted_queries:
            # Expand the right boundary to include all logs with time <= q
            while right < len(sorted_logs) and sorted_logs[right][1] <= q:
                server_id = sorted_logs[right][0]
                if server_counts[server_id] == 0:
                    unique += 1
                server_counts[server_id] += 1
                right += 1
            
            # Shrink the left boundary to exclude logs with time < (q - x)
            while left < len(sorted_logs) and sorted_logs[left][1] < (q - x):
                server_id = sorted_logs[left][0]
                server_counts[server_id] -= 1
                if server_counts[server_id] == 0:
                    unique -= 1
                left += 1
            
            # The answer is total servers minus those active in the window
            ans[original_idx] = n - unique
        
        return ans