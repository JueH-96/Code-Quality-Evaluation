class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda log: log[1])
        sorted_queries = sorted([(q, idx) for idx, q in enumerate(queries)])
        
        counts = [0] * (n + 1)
        active = 0
        left = 0
        right = 0
        m = len(logs)
        ans = [0] * len(queries)
        
        for q, idx in sorted_queries:
            while right < m and logs[right][1] <= q:
                server_id = logs[right][0]
                counts[server_id] += 1
                if counts[server_id] == 1:
                    active += 1
                right += 1
            while left < m and logs[left][1] < q - x:
                server_id = logs[left][0]
                counts[server_id] -= 1
                if counts[server_id] == 0:
                    active -= 1
                left += 1
            ans[idx] = n - active
        
        return ans