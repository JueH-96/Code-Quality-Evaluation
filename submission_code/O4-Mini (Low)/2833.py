from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda l: l[1])
        # Prepare queries with original indices and sort by query time
        queries_with_idx = sorted([(q, i) for i, q in enumerate(queries)], key=lambda x: x[0])
        
        ans = [0] * len(queries)
        counts = [0] * (n + 1)  # counts per server ID
        active = 0  # number of servers with at least one request in current window
        
        left = 0   # pointer to remove outdated logs
        right = 0  # pointer to add new logs
        
        for q_time, q_idx in queries_with_idx:
            window_start = q_time - x
            
            # Add logs up to q_time
            while right < len(logs) and logs[right][1] <= q_time:
                server_id = logs[right][0]
                if counts[server_id] == 0:
                    active += 1
                counts[server_id] += 1
                right += 1
            
            # Remove logs before window_start
            while left < len(logs) and logs[left][1] < window_start:
                server_id = logs[left][0]
                counts[server_id] -= 1
                if counts[server_id] == 0:
                    active -= 1
                left += 1
            
            # Servers with zero requests = total servers - active servers
            ans[q_idx] = n - active
        
        return ans