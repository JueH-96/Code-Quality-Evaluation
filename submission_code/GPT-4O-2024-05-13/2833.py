class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Prepare the result array
        result = []
        
        # Iterate over each query
        for query in queries:
            start_time = query - x
            end_time = query
            
            # Use a set to track servers that received requests in the interval
            active_servers = set()
            
            # Iterate over logs to find active servers in the given interval
            for server_id, time in logs:
                if start_time <= time <= end_time:
                    active_servers.add(server_id)
                elif time > end_time:
                    break
            
            # Calculate the number of servers that did not receive any requests
            inactive_servers_count = n - len(active_servers)
            result.append(inactive_servers_count)
        
        return result