from collections import defaultdict
from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort the logs based on their time
        logs.sort(key=lambda log: log[1])
        
        # Sort the queries along with their original indices
        sorted_queries = sorted([(q, i) for i, q in enumerate(queries)])
        
        # Initialize variables for the sliding window
        left = 0
        right = 0
        freq = defaultdict(int)
        current_count = 0
        res = [0] * len(queries)
        
        for q_val, idx in sorted_queries:
            start = q_val - x
            
            # Expand the right pointer to include all logs up to q_val
            while right < len(logs) and logs[right][1] <= q_val:
                sid = logs[right][0]
                if freq[sid] == 0:
                    current_count += 1
                freq[sid] += 1
                right += 1
            
            # Contract the left pointer to exclude logs before start
            while left < len(logs) and logs[left][1] < start:
                sid = logs[left][0]
                freq[sid] -= 1
                if freq[sid] == 0:
                    current_count -= 1
                left += 1
            
            # Calculate the number of servers with no requests in the current window
            res[idx] = n - current_count
        
        return res