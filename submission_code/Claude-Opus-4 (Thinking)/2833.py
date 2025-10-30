class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        import bisect
        
        # Create a list of (time, server_id) and sort by time
        time_server = [(time, server_id) for server_id, time in logs]
        time_server.sort()
        
        # Extract just the times for binary search
        times = [t for t, _ in time_server]
        
        result = []
        
        for q in queries:
            start_time = q - x
            end_time = q
            
            # Find the range of logs within [start_time, end_time]
            left = bisect.bisect_left(times, start_time)
            right = bisect.bisect_right(times, end_time)
            
            # Count unique servers in this range
            servers_with_requests = set()
            for i in range(left, right):
                servers_with_requests.add(time_server[i][1])
            
            # Number of servers without requests
            servers_without_requests = n - len(servers_with_requests)
            result.append(servers_without_requests)
        
        return result