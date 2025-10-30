import collections
from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # 1. Sort logs by their timestamp. This is crucial for the two-pointer approach.
        # logs[i] = [server_id, time]
        logs.sort(key=lambda item: item[1])

        # 2. Pair queries with their original indices and sort them by query time.
        # This allows processing queries in time order while preserving the original output order.
        indexed_queries = []
        for i, q_time in enumerate(queries):
            indexed_queries.append((q_time, i))
        indexed_queries.sort()

        # 3. Initialize the result array to store answers for each query.
        results = [0] * len(queries)

        # 4. Data structures for the sliding window:
        #    server_activity_counts: Stores the number of requests for each server_id
        #                            within the current time window.
        #    active_server_count: Tracks the number of unique servers that have at least
        #                         one request in the current window.
        server_activity_counts = collections.defaultdict(int)
        active_server_count = 0

        # 5. Initialize two pointers for the logs array.
        left_ptr = 0  # Points to the start of the current window (logs to potentially remove)
        right_ptr = 0 # Points to the end of the current window (logs to potentially add)

        # 6. Process each query in sorted order.
        for query_time, original_index in indexed_queries:
            # Define the time window for the current query.
            start_time = query_time - x
            end_time = query_time

            # Phase 1: Expand the window from the right.
            # Add logs whose timestamps are within or before the end of the current query's window.
            while right_ptr < len(logs) and logs[right_ptr][1] <= end_time:
                server_id = logs[right_ptr][0]
                # If this server was not active in the window (count was 0), increment active_server_count.
                if server_activity_counts[server_id] == 0:
                    active_server_count += 1
                server_activity_counts[server_id] += 1
                right_ptr += 1

            # Phase 2: Shrink the window from the left.
            # Remove logs whose timestamps are strictly before the start of the current query's window.
            while left_ptr < len(logs) and logs[left_ptr][1] < start_time:
                server_id = logs[left_ptr][0]
                server_activity_counts[server_id] -= 1
                # If this server is no longer active in the window (count becomes 0), decrement active_server_count.
                if server_activity_counts[server_id] == 0:
                    active_server_count -= 1
                left_ptr += 1
            
            # Calculate the result for the current query.
            # The total servers are 'n'. 'active_server_count' are servers that received requests.
            # The difference is the servers that did NOT receive requests.
            results[original_index] = n - active_server_count

        return results