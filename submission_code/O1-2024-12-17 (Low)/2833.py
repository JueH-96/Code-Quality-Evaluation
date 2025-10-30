class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda e: e[1])
        
        # Prepare queries alongside their original indices, then sort by query time
        indexed_queries = sorted([(q, i) for i, q in enumerate(queries)], key=lambda x: x[0])
        
        # Frequency array to keep track of how many logs are in the window for each server
        freq = [0] * (n + 1)  # server ids are 1-based
        count_active = 0      # how many different servers have freq > 0
        left = 0              # left pointer for the logs
        right = 0             # right pointer for the logs
        L = len(logs)
        
        results = [0] * len(queries)
        
        for q_time, idx in indexed_queries:
            # Extend window's right edge: include logs with time <= q_time
            while right < L and logs[right][1] <= q_time:
                server_id, _ = logs[right]
                if freq[server_id] == 0:
                    count_active += 1
                freq[server_id] += 1
                right += 1
            
            # Shrink window's left edge: exclude logs with time < q_time - x
            start_time = q_time - x
            while left < right and logs[left][1] < start_time:
                server_id, _ = logs[left]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    count_active -= 1
                left += 1
            
            # Number of servers with no requests in [q_time - x, q_time]
            results[idx] = n - count_active
        
        return results