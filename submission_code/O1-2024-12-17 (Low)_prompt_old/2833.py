class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda x: x[1])
        
        # Pair each query with its original index and sort by query time
        indexed_queries = sorted([(q, i) for i, q in enumerate(queries)], key=lambda x: x[0])
        
        # We'll use two pointers (left, right) to maintain the sliding window of log times
        left = 0
        right = 0
        
        # Frequency array to count how many requests each server currently has in the window
        freq = [0] * (n + 1)
        distinct = 0  # Number of servers with at least one request in the current window
        
        # Result array (same size as queries) to fill in correct order
        result = [0] * len(queries)
        
        # Total number of logs
        total_logs = len(logs)
        
        # Process queries in ascending order of their times
        for q_time, q_index in indexed_queries:
            # Expand the right pointer to include all logs with time <= q_time
            while right < total_logs and logs[right][1] <= q_time:
                server_id = logs[right][0]
                # If this server is just becoming non-zero frequency
                if freq[server_id] == 0:
                    distinct += 1
                freq[server_id] += 1
                right += 1
            
            # Shrink the left pointer to exclude logs with time < q_time - x
            lower_bound = q_time - x
            while left < total_logs and logs[left][1] < lower_bound:
                server_id = logs[left][0]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    distinct -= 1
                left += 1
            
            # Now 'distinct' is the number of servers that have at least 1 log in [q_time - x, q_time]
            # So the number of servers with 0 requests in that interval is n - distinct
            result[q_index] = n - distinct
        
        return result