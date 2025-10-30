from sortedcontainers import SortedList

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort the logs by time
        logs.sort(key=lambda log: log[1])
        
        # Sort the queries by time and keep track of their original indices
        sorted_queries = sorted((query, i) for i, query in enumerate(queries))
        
        result = [0] * len(queries)
        server_activity = [0] * n  # Tracks activity of each server
        active_servers = SortedList()  # Stores active servers' indices
        
        query_index = 0
        log_index = 0
        
        for query, original_index in sorted_queries:
            # Add logs that fall within the current query's time window
            while log_index < len(logs) and logs[log_index][1] <= query:
                server_id = logs[log_index][0] - 1
                if server_activity[server_id] == 0:
                    active_servers.add(server_id)
                server_activity[server_id] += 1
                log_index += 1
            
            # Remove logs that fall outside the current query's time window
            while active_servers and server_activity[active_servers[0]] > 0 and logs[query_index][1] + x < query:
                server_id = active_servers[0]
                server_activity[server_id] -= 1
                if server_activity[server_id] == 0:
                    active_servers.remove(server_id)
                query_index += 1
            
            # Calculate the number of inactive servers
            result[original_index] = n - len(active_servers)
        
        return result