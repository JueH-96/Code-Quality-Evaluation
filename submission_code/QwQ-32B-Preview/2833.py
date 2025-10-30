from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort the logs by time
        logs_sorted = sorted(logs, key=lambda x: x[1])
        
        # Prepare queries with start and end times and original indices
        queries_with_idx = [(query - x, query, idx) for idx, query in enumerate(queries)]
        # Sort queries by the start time
        queries_by_start = sorted(queries_with_idx, key=lambda x: x[0])
        
        # Initialize pointers
        left = 0
        right = 0
        # Initialize server count array
        server_count = [0] * (n + 1)
        # Counter for servers in the current window
        in_window = 0
        # Result list
        result = [0] * len(queries)
        
        # Process each query in sorted order
        for query_start, query_end, original_idx in queries_by_start:
            # Remove logs that are before the current query's start time
            while left < len(logs_sorted) and logs_sorted[left][1] < query_start:
                server = logs_sorted[left][0]
                server_count[server] -= 1
                if server_count[server] == 0:
                    in_window -= 1
                left += 1
            # Add logs that are within the current query's end time
            while right < len(logs_sorted) and logs_sorted[right][1] <= query_end:
                server = logs_sorted[right][0]
                if server_count[server] == 0:
                    in_window += 1
                server_count[server] += 1
                right += 1
            # Calculate the number of servers not receiving any requests
            result[original_idx] = n - in_window
        
        return result