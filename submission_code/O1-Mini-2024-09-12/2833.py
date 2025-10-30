import bisect
from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Initialize last_log with 0 for servers that have no logs
        last_log = [0] * (n + 1)
        
        # Update last_log with the latest request time for each server
        for server_id, time in logs:
            if time > last_log[server_id]:
                last_log[server_id] = time
        
        # Extract and sort the last log times
        sorted_last_logs = sorted(last_log[1:])
        
        # Process each query
        result = []
        for q in queries:
            threshold = q - x
            # Find the number of servers with last_log_time < threshold
            count = bisect.bisect_left(sorted_last_logs, threshold)
            result.append(count)
        
        return result