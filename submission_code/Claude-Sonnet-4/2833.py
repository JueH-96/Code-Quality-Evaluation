class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        result = []
        
        for query in queries:
            start_time = query - x
            end_time = query
            
            # Find the range of logs that fall within [start_time, end_time]
            # Use binary search to find the first log with time >= start_time
            left = 0
            right = len(logs)
            while left < right:
                mid = (left + right) // 2
                if logs[mid][1] >= start_time:
                    right = mid
                else:
                    left = mid + 1
            start_idx = left
            
            # Use binary search to find the first log with time > end_time
            left = 0
            right = len(logs)
            while left < right:
                mid = (left + right) // 2
                if logs[mid][1] > end_time:
                    right = mid
                else:
                    left = mid + 1
            end_idx = left
            
            # Count unique servers in the range [start_idx, end_idx)
            active_servers = set()
            for i in range(start_idx, end_idx):
                active_servers.add(logs[i][0])
            
            # Number of servers that did NOT receive requests
            inactive_count = n - len(active_servers)
            result.append(inactive_count)
        
        return result