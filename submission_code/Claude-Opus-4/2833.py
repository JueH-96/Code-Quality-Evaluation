class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Create list of (query_value, original_index) and sort by query value
        indexed_queries = [(q, i) for i, q in enumerate(queries)]
        indexed_queries.sort()
        
        result = [0] * len(queries)
        
        left = 0
        right = 0
        server_count = {}  # Count of requests per server in current window
        
        for query, original_idx in indexed_queries:
            start_time = query - x
            end_time = query
            
            # Move right pointer to include all logs with time <= end_time
            while right < len(logs) and logs[right][1] <= end_time:
                server_id = logs[right][0]
                server_count[server_id] = server_count.get(server_id, 0) + 1
                right += 1
            
            # Move left pointer to exclude logs with time < start_time
            while left < len(logs) and logs[left][1] < start_time:
                server_id = logs[left][0]
                server_count[server_id] -= 1
                if server_count[server_id] == 0:
                    del server_count[server_id]
                left += 1
            
            # Number of servers that didn't receive any requests
            servers_with_requests = len(server_count)
            result[original_idx] = n - servers_with_requests
        
        return result