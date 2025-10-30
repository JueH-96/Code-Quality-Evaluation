class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from bisect import bisect_left, bisect_right
        
        # Group logs by server_id
        server_logs = {}
        for server_id, time in logs:
            if server_id not in server_logs:
                server_logs[server_id] = []
            server_logs[server_id].append(time)
        
        # Sort the logs for each server
        for server_id in server_logs:
            server_logs[server_id].sort()
        
        result = []
        for query in queries:
            window_start = query - x
            window_end = query
            
            inactive_count = 0
            for server_id in range(1, n + 1):
                # Check if server is inactive
                if server_id not in server_logs:
                    # If server has no logs, it's inactive
                    inactive_count += 1
                else:
                    times = server_logs[server_id]
                    # Find the range of logs within the time window
                    left = bisect_left(times, window_start)
                    right = bisect_right(times, window_end)
                    
                    # If left == right, there are no logs in the time window
                    if left == right:
                        inactive_count += 1
            
            result.append(inactive_count)
        
        return result