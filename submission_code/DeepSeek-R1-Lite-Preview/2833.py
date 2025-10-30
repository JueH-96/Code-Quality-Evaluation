from typing import List
import bisect

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Initialize latest request times for each server to 0
        latest_times = [0] * (n + 1)
        
        # Update latest request times based on logs
        for server_id, time in logs:
            if time > latest_times[server_id]:
                latest_times[server_id] = time
        
        # Collect and sort the latest request times for all servers
        server_latest_times = latest_times[1:n+1]
        server_latest_times.sort()
        
        # Prepare the result list
        result = []
        
        # For each query, compute threshold and count servers with latest_time < threshold
        for q in queries:
            threshold = q - x
            count = bisect.bisect_left(server_latest_times, threshold)
            result.append(count)
        
        return result