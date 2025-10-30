from typing import List
from collections import defaultdict
from bisect import bisect_left

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        server_logs = defaultdict(list)
        for server_id, time in logs:
            server_logs[server_id].append(time)
        
        for server_id in server_logs:
            server_logs[server_id].sort()
        
        result = []
        for query in queries:
            count = 0
            for server_id in range(1, n+1):
                if server_id not in server_logs or server_logs[server_id][bisect_left(server_logs[server_id], query - x)] > query - x:
                    count += 1
            result.append(count)
        
        return result