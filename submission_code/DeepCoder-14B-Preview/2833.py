import bisect

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort the logs based on the time of the requests
        logs_sorted = sorted(logs, key=lambda x: x[1])
        times = [log[1] for log in logs_sorted]
        servers = [log[0] for log in logs_sorted]
        
        results = []
        for q in queries:
            L = q - x
            R = q
            # Find the left and right indices using binary search
            left = bisect.bisect_left(times, L)
            right_idx = bisect.bisect_right(times, R)
            # Extract the server IDs in the current time window
            servers_in_window = servers[left:right_idx]
            # Count the number of unique servers
            unique_servers = len(set(servers_in_window))
            # Calculate the result for this query
            results.append(n - unique_servers)
        return results