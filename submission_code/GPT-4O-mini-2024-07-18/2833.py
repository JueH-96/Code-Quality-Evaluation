from typing import List
from collections import defaultdict

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Create a dictionary to store the times when each server received requests
        server_requests = defaultdict(list)
        
        # Fill the server_requests dictionary with the logs
        for server_id, time in logs:
            server_requests[server_id].append(time)
        
        # Prepare the result array
        result = []
        
        # Process each query
        for query in queries:
            start_time = query - x
            end_time = query
            
            # Count servers that did not receive any requests in the interval [start_time, end_time]
            active_servers = set()
            for server_id in range(1, n + 1):
                if server_id in server_requests:
                    # Check if this server received requests in the given time interval
                    for request_time in server_requests[server_id]:
                        if start_time <= request_time <= end_time:
                            active_servers.add(server_id)
                            break
            
            # The number of servers that did not receive any requests
            inactive_count = n - len(active_servers)
            result.append(inactive_count)
        
        return result