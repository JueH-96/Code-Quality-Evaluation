from typing import List
from collections import defaultdict

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Prepare result array
        result = []
        
        # Iterate over each query
        for query in queries:
            # Calculate the time interval
            start_time = query - x
            end_time = query
            
            # Set to track servers that received requests in the interval
            active_servers = set()
            
            # Iterate over logs and find active servers in the interval
            for server_id, time in logs:
                if start_time <= time <= end_time:
                    active_servers.add(server_id)
                elif time > end_time:
                    # Since logs are sorted by time, we can break early
                    break
            
            # Calculate the number of servers that did not receive any requests
            inactive_servers_count = n - len(active_servers)
            result.append(inactive_servers_count)
        
        return result