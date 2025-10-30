from typing import List
from collections import defaultdict
from bisect import bisect_left, bisect_right

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Map each query to its index for sorting and result mapping
        sorted_queries = sorted((q, i) for i, q in enumerate(queries))
        
        # Initialize variables
        result = [0] * len(queries)
        server_count = defaultdict(int)
        active_logs = 0
        log_idx = 0
        
        for q, original_idx in sorted_queries:
            # Add logs that fall within the current query window
            while log_idx < len(logs) and logs[log_idx][1] <= q:
                server_id = logs[log_idx][0]
                server_count[server_id] += 1
                if server_count[server_id] == 1:
                    active_logs += 1
                log_idx += 1
            
            # Remove logs that fall outside the current query window
            start_time = q - x
            while log_idx > 0 and logs[log_idx - 1][1] < start_time:
                log_idx -= 1
                server_id = logs[log_idx][0]
                server_count[server_id] -= 1
                if server_count[server_id] == 0:
                    active_logs -= 1
            
            # Calculate the number of inactive servers
            result[original_idx] = n - active_logs
        
        return result