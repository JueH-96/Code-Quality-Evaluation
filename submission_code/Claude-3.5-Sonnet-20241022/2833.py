class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda x: x[1])
        
        # Create list of query indices sorted by time
        query_indices = sorted(range(len(queries)), key=lambda i: queries[i])
        
        result = [0] * len(queries)
        active_servers = set()
        
        left = 0
        right = 0
        
        # Process each query in sorted order
        for query_idx in query_indices:
            query_time = queries[query_idx]
            window_start = query_time - x
            window_end = query_time
            
            # Add servers that fall in the window
            while right < len(logs) and logs[right][1] <= window_end:
                active_servers.add(logs[right][0])
                right += 1
            
            # Remove servers that are before window start
            while left < right and logs[left][1] < window_start:
                # Only remove if no other request from same server in window
                curr_server = logs[left][0]
                can_remove = True
                
                # Check if server has another request in window
                for i in range(left + 1, right):
                    if logs[i][0] == curr_server and logs[i][1] >= window_start:
                        can_remove = False
                        break
                
                if can_remove:
                    active_servers.discard(curr_server)
                left += 1
            
            # Calculate servers with no requests in window
            result[query_idx] = n - len(active_servers)
            
        return result