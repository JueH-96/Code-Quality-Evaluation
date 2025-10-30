from collections import defaultdict
import bisect

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Create a dictionary to map each server to its list of request times
        server_logs = defaultdict(list)
        for server_id, time in logs:
            server_logs[server_id].append(time)
        
        # Sort the request times for each server
        for server_id in server_logs:
            server_logs[server_id].sort()
        
        # Initialize the result array
        result = []
        
        # Iterate over each query
        for q in queries:
            start = q - x
            end = q
            active_servers = set()
            # Iterate over each server
            for server_id in server_logs:
                # Find the first index where time >= start
                left = bisect.bisect_left(server_logs[server_id], start)
                # Find the first index where time > end
                right = bisect.bisect_right(server_logs[server_id], end)
                # If there is at least one request in the interval, add to active_servers
                if right > left:
                    active_servers.add(server_id)
            # The number of servers not in active_servers is n - len(active_servers)
            result.append(n - len(active_servers))
        
        return result