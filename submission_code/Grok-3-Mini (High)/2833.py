from typing import List
from collections import Counter

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs_sorted = sorted(logs, key=lambda log: log[1])
        sorted_queries = sorted(enumerate(queries), key=lambda x: x[1])
        server_count = Counter()
        unique_servers = 0
        idx_left = 0
        idx_right = -1
        len_logs = len(logs_sorted)
        result = [0] * len(queries)
        for orig_idx, query_time in sorted_queries:
            curr_lower = query_time - x
            curr_upper = query_time
            while idx_right + 1 < len_logs and logs_sorted[idx_right + 1][1] <= curr_upper:
                idx_right += 1
                server_id = logs_sorted[idx_right][0]
                server_count[server_id] += 1
                if server_count[server_id] == 1:
                    unique_servers += 1
            while idx_left <= idx_right and logs_sorted[idx_left][1] < curr_lower:
                server_id = logs_sorted[idx_left][0]
                server_count[server_id] -= 1
                if server_count[server_id] == 0:
                    unique_servers -= 1
                idx_left += 1
            no_request = n - unique_servers
            result[orig_idx] = no_request
        return result