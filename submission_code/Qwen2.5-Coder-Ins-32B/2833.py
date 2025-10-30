from typing import List
from collections import defaultdict
from bisect import bisect_left, bisect_right

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Store the indices of logs for each server
        server_logs = defaultdict(list)
        for i, (server_id, time) in enumerate(logs):
            server_logs[server_id].append(i)
        
        # Sort queries and keep track of original indices
        sorted_queries = sorted((query, i) for i, query in enumerate(queries))
        
        # Result array
        result = [0] * len(queries)
        
        # Two pointers for the sliding window
        left = 0
        right = 0
        active_servers = set()
        
        for query, original_index in sorted_queries:
            start_time = query - x
            
            # Move the right pointer to include all logs up to the current query time
            while right < len(logs) and logs[right][1] <= query:
                server_id = logs[right][1]
                active_servers.add(logs[right][0])
                right += 1
            
            # Move the left pointer to exclude all logs before the start time
            while left < right and logs[left][1] < start_time:
                server_id = logs[left][0]
                # If this is the only log for this server in the current window, remove it
                if bisect_left(server_logs[server_id], left) == bisect_right(server_logs[server_id], left) - 1:
                    active_servers.remove(server_id)
                left += 1
            
            # The number of servers that did not receive any requests
            result[original_index] = n - len(active_servers)
        
        return result