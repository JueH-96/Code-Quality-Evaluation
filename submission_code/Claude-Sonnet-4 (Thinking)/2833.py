class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Create a list of (query_value, original_index) and sort by query_value
        queries_with_idx = sorted((q, i) for i, q in enumerate(queries))
        
        result = [0] * len(queries)
        
        left_ptr = 0
        right_ptr = 0
        servers_count = {}
        
        for q, original_idx in queries_with_idx:
            left_time = q - x
            right_time = q
            
            # Move right_ptr to include all logs with time <= right_time
            while right_ptr < len(logs) and logs[right_ptr][1] <= right_time:
                server_id = logs[right_ptr][0]
                servers_count[server_id] = servers_count.get(server_id, 0) + 1
                right_ptr += 1
            
            # Move left_ptr to exclude all logs with time < left_time
            while left_ptr < len(logs) and logs[left_ptr][1] < left_time:
                server_id = logs[left_ptr][0]
                servers_count[server_id] -= 1
                if servers_count[server_id] == 0:
                    del servers_count[server_id]
                left_ptr += 1
            
            result[original_idx] = n - len(servers_count)
        
        return result