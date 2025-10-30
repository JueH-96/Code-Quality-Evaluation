from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort the logs by time to facilitate efficient range queries
        sorted_logs = sorted(logs, key=lambda log: log[1])
        times = [log[1] for log in sorted_logs]
        server_ids = [log[0] for log in sorted_logs]
        
        # Prepare the queries with their original indices and computed ranges
        processed_queries = []
        for idx, q in enumerate(queries):
            L = q - x
            R = q
            # Find the left and right indices using binary search
            left = bisect_left(times, L)
            right = bisect_right(times, R)
            # The window is [left, right - 1]
            processed_queries.append((left, right - 1, idx))
        
        # Determine the block size for Mo's algorithm
        block_size = int(len(sorted_logs) ** 0.5) if len(sorted_logs) > 0 else 1
        
        # Custom comparison function for sorting queries in Mo's order
        def mo_cmp(query):
            l, r, idx = query
            block = l // block_size
            # Even blocks sorted by r ascending, odd blocks by r descending
            return (block, r if block % 2 == 0 else -r)
        
        # Sort the queries using Mo's algorithm order
        processed_queries.sort(key=mo_cmp)
        
        # Initialize variables for Mo's algorithm
        current_l = 0
        current_r = 0
        frequency = defaultdict(int)
        current_count = 0
        ans = [0] * len(processed_queries)
        
        # Process each query using Mo's algorithm
        for q in processed_queries:
            l, r, idx = q
            
            # Expand to the left if current_l is greater than l
            while current_l > l:
                current_l -= 1
                sid = server_ids[current_l]
                frequency[sid] += 1
                if frequency[sid] == 1:
                    current_count += 1
            
            # Expand to the right if current_r is less than or equal to r
            while current_r <= r and current_r < len(server_ids):
                sid = server_ids[current_r]
                frequency[sid] += 1
                if frequency[sid] == 1:
                    current_count += 1
                current_r += 1
            
            # Contract current_l to the right if current_l is less than l
            while current_l < l:
                sid = server_ids[current_l]
                frequency[sid] -= 1
                if frequency[sid] == 0:
                    current_count -= 1
                current_l += 1
            
            # Contract current_r to the left if current_r is greater than r + 1
            while current_r > r + 1:
                current_r -= 1
                sid = server_ids[current_r]
                frequency[sid] -= 1
                if frequency[sid] == 0:
                    current_count -= 1
            
            # Store the current count of distinct servers for this query
            ans[idx] = current_count
        
        # Calculate the final answer for each query
        final_ans = [n - ans[i] for i in range(len(ans))]
        
        return final_ans