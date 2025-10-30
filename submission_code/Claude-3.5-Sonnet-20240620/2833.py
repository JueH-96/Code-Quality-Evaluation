class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Sort queries and keep track of original indices
        sorted_queries = sorted(enumerate(queries), key=lambda q: q[1])
        
        result = [0] * len(queries)
        active_servers = {}
        log_index = 0
        inactive_count = n
        
        for query_index, query_time in sorted_queries:
            # Add servers that received requests before or at query_time
            while log_index < len(logs) and logs[log_index][1] <= query_time:
                server_id, request_time = logs[log_index]
                if server_id not in active_servers:
                    inactive_count -= 1
                active_servers[server_id] = request_time
                log_index += 1
            
            # Remove servers whose requests are outside the time window
            for server_id in list(active_servers.keys()):
                if active_servers[server_id] <= query_time - x:
                    del active_servers[server_id]
                    inactive_count += 1
            
            result[query_index] = inactive_count
        
        return result