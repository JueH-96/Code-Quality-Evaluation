class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by their time
        logs.sort(key=lambda log: log[1])
        
        # Prepare sorted queries with (end, start, original_index)
        sorted_queries = sorted([(q, q - x, idx) for idx, q in enumerate(queries)], key=lambda q: q[0])
        
        m = len(queries)
        res = [0] * m
        counter = [0] * (n + 1)  # 1-based server IDs
        active_count = 0
        left = 0
        right = 0
        
        for current_e, current_s, original_idx in sorted_queries:
            # Expand the right pointer to include all logs with time <= current_e
            while right < len(logs) and logs[right][1] <= current_e:
                server_id = logs[right][0]
                if counter[server_id] == 0:
                    active_count += 1
                counter[server_id] += 1
                right += 1
            
            # Expand the left pointer to exclude all logs with time < current_s
            while left < len(logs) and logs[left][1] < current_s:
                server_id = logs[left][0]
                counter[server_id] -= 1
                if counter[server_id] == 0:
                    active_count -= 1
                left += 1
            
            # Save the result for the current query
            res[original_idx] = n - active_count
        
        return res