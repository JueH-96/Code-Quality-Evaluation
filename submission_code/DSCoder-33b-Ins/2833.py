class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda x: x[1])
        queries = sorted([(t, i) for i, t in enumerate(queries)])
        servers = [0] * (n + 1)
        unavailable = set()
        j = 0
        for t, i in queries:
            while j < len(logs) and logs[j][1] <= t:
                server_id = logs[j][0]
                servers[server_id] += 1
                if servers[server_id] == 1:
                    unavailable.remove(server_id)
                j += 1
            while j < len(logs) and logs[j][1] < t - x + 1:
                server_id = logs[j][0]
                servers[server_id] -= 1
                if servers[server_id] == 0:
                    unavailable.add(server_id)
                j += 1
            queries[i] = len(unavailable)
        return queries