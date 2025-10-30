from collections import defaultdict
from bisect import bisect_left, bisect_right

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Create a dictionary to store the times for each server
        server_times = defaultdict(list)
        for server_id, time in logs:
            server_times[server_id].append(time)
        
        # Sort queries and keep track of original indices
        sorted_queries = sorted((q, i) for i, q in enumerate(queries))
        
        # Initialize the result array
        result = [0] * len(queries)
        
        # Initialize a set to keep track of active servers
        active_servers = set()
        
        # Initialize a pointer for the logs
        log_ptr = 0
        
        for q, original_index in sorted_queries:
            # Add servers that have requests in the interval [q - x, q]
            while log_ptr < len(logs) and logs[log_ptr][1] <= q:
                server_id, time = logs[log_ptr]
                if time >= q - x:
                    active_servers.add(server_id)
                log_ptr += 1
            
            # Remove servers that have requests outside the interval [q - x, q]
            for server_id in list(active_servers):
                times = server_times[server_id]
                left = bisect_left(times, q - x)
                right = bisect_right(times, q)
                if left == right:
                    active_servers.remove(server_id)
            
            # The number of servers that did not receive any requests
            result[original_index] = n - len(active_servers)
        
        return result