from collections import defaultdict
from sortedcontainers import SortedList

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda log: log[1])
        queries = sorted((q, i) for i, q in enumerate(queries))
        ans = [0] * len(queries)
        server_times = defaultdict(SortedList)
        active_servers = SortedList()
        
        for q, i in queries:
            while logs and logs[0][1] <= q:
                server_id, time = logs[0]
                server_times[server_id].add(time)
                active_servers.add(server_id)
                logs.pop(0)
            
            while active_servers and active_servers[0] < q - x:
                server_id = active_servers[0]
                server_times[server_id].remove(q - x)
                if not server_times[server_id]:
                    active_servers.remove(server_id)
            
            ans[i] = n - len(active_servers)
        
        return ans