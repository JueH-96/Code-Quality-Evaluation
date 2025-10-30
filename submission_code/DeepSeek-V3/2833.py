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
        
        # Prepare the result list
        result = []
        
        # For each query, determine the number of servers that did not receive any requests in the interval [q - x, q]
        for q in queries:
            start = q - x
            end = q
            count = 0
            # Iterate through all servers
            for server_id in range(1, n+1):
                times = server_logs.get(server_id, [])
                # Find the number of requests in the interval [start, end]
                left = bisect.bisect_left(times, start)
                right = bisect.bisect_right(times, end)
                if right - left == 0:
                    count += 1
            result.append(count)
        
        return result