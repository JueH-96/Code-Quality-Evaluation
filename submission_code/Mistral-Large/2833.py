from typing import List
from collections import defaultdict
import heapq

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Dictionary to store the logs for each server
        server_logs = defaultdict(list)

        # Populate the server_logs dictionary
        for server_id, time in logs:
            server_logs[server_id].append(time)

        # Sort the logs for each server
        for server_id in server_logs:
            server_logs[server_id].sort()

        # Min-heap to keep track of the next log time for each server
        min_heap = []
        for server_id in server_logs:
            heapq.heappush(min_heap, (server_logs[server_id][0], server_id, 0))

        # Result array
        result = []

        # Process each query
        for query in queries:
            start_time = query - x
            end_time = query

            # Temporary set to keep track of servers that have logs in the current query range
            active_servers = set()

            # Process the min-heap
            while min_heap and min_heap[0][0] <= end_time:
                time, server_id, index = heapq.heappop(min_heap)
                if time >= start_time:
                    active_servers.add(server_id)
                if index + 1 < len(server_logs[server_id]):
                    next_time = server_logs[server_id][index + 1]
                    heapq.heappush(min_heap, (next_time, server_id, index + 1))

            # Calculate the number of servers that did not receive any requests during the query range
            inactive_servers = n - len(active_servers)
            result.append(inactive_servers)

        return result