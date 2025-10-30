class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda x: x[1])
        queries = sorted((q, i) for i, q in enumerate(queries))
        ans = [0] * len(queries)
        i = j = 0
        count = {}
        for q, idx in queries:
            while j < len(logs) and logs[j][1] <= q:
                server, time = logs[j]
                count[server] = count.get(server, 0) + 1
                j += 1
            while i < len(logs) and logs[i][1] < q - x + 1:
                server, time = logs[i]
                count[server] -= 1
                if count[server] == 0:
                    del count[server]
                i += 1
            ans[idx] = n - len(count)
        return ans