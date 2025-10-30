from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda log: log[1])
        sorted_queries = sorted([(q, idx) for idx, q in enumerate(queries)])
        freq = [0] * (n + 1)
        distinct = 0
        left = 0
        right = 0
        ans = [0] * len(queries)
        
        for q, idx in sorted_queries:
            low_bound = q - x
            while right < len(logs) and logs[right][1] <= q:
                server_id = logs[right][0]
                if freq[server_id] == 0:
                    distinct += 1
                freq[server_id] += 1
                right += 1
            
            while left < len(logs) and logs[left][1] < low_bound:
                server_id = logs[left][0]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    distinct -= 1
                left += 1
            
            ans[idx] = n - distinct
        
        return ans