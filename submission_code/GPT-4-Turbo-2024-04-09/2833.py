class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from collections import defaultdict
        
        # Dictionary to store the times each server received requests
        server_requests = defaultdict(list)
        for server_id, time in logs:
            server_requests[server_id].append(time)
        
        # Sort the request times for each server for efficient interval checking
        for server_id in server_requests:
            server_requests[server_id].sort()
        
        # Function to count how many times a server received requests in a given interval
        def count_requests_in_interval(times, start, end):
            # Binary search to find the count of requests in the interval [start, end]
            from bisect import bisect_left, bisect_right
            left_idx = bisect_left(times, start)
            right_idx = bisect_right(times, end) - 1
            if left_idx <= right_idx and right_idx < len(times) and left_idx >= 0:
                return right_idx - left_idx + 1
            return 0
        
        result = []
        
        # Process each query
        for query in queries:
            start_time = query - x
            end_time = query
            count_inactive_servers = 0
            
            # Check each server if it received requests in the interval [start_time, end_time]
            for server_id in range(1, n + 1):
                if server_id in server_requests:
                    if count_requests_in_interval(server_requests[server_id], start_time, end_time) == 0:
                        count_inactive_servers += 1
                else:
                    # This server never received any requests
                    count_inactive_servers += 1
            
            result.append(count_inactive_servers)
        
        return result