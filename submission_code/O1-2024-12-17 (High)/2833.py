class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Convert logs into a list of (time, server_id) pairs and sort by time
        requests = [(t, s) for s, t in logs]
        requests.sort(key=lambda x: x[0])
        
        # Prepare queries alongside their original indices, then sort by query time
        indexed_queries = sorted([(q, i) for i, q in enumerate(queries)], key=lambda x: x[0])
        
        # Frequency array to track how many requests each server has in the current window
        freq = [0] * (n + 1)
        distinct_in_window = 0
        
        left = 0
        right = 0
        total_logs = len(requests)
        
        # This will hold our answers; we'll fill it in the order of the queries' original indices
        ans = [0] * len(queries)
        
        # Process queries in ascending order of time
        for q_time, idx in indexed_queries:
            # Expand the window to include all requests up to time q_time
            while right < total_logs and requests[right][0] <= q_time:
                t, s = requests[right]
                if freq[s] == 0:
                    distinct_in_window += 1
                freq[s] += 1
                right += 1
            
            # Shrink the window to remove requests before (q_time - x)
            window_start = q_time - x
            while left < total_logs and requests[left][0] < window_start:
                t, s = requests[left]
                freq[s] -= 1
                if freq[s] == 0:
                    distinct_in_window -= 1
                left += 1
            
            # Number of servers with zero requests is total servers minus the distinct count
            ans[idx] = n - distinct_in_window
        
        return ans