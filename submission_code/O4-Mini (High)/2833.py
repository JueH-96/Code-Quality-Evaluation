from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by timestamp
        logs_sorted = sorted(logs, key=lambda entry: entry[1])
        # Pair each query with its original index and sort by query time
        sorted_queries = sorted((q, i) for i, q in enumerate(queries))
        
        ans = [0] * len(queries)
        cnt = [0] * (n + 1)   # cnt[s] = number of logs of server s in the current window
        zero_count = n        # number of servers with cnt == 0
        
        left = 0
        right = 0
        L = len(logs_sorted)
        
        for q, qi in sorted_queries:
            # Expand right pointer to include logs with time <= q
            while right < L and logs_sorted[right][1] <= q:
                srv = logs_sorted[right][0]
                cnt[srv] += 1
                if cnt[srv] == 1:
                    zero_count -= 1
                right += 1
            
            # Move left pointer to exclude logs with time < q - x
            start_time = q - x
            while left < right and logs_sorted[left][1] < start_time:
                srv = logs_sorted[left][0]
                cnt[srv] -= 1
                if cnt[srv] == 0:
                    zero_count += 1
                left += 1
            
            # At this point, zero_count is the number of servers with no logs in [q-x, q]
            ans[qi] = zero_count
        
        return ans