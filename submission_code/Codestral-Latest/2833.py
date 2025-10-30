class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from collections import defaultdict
        from sortedcontainers import SortedList

        # Dictionary to store the requests for each server
        server_requests = defaultdict(SortedList)

        # Populate the server_requests dictionary
        for server_id, time in logs:
            server_requests[server_id].add(time)

        # Function to count servers with no requests in the given interval
        def count_no_requests(interval_start, interval_end):
            count = 0
            for server_id in range(1, n + 1):
                requests = server_requests[server_id]
                # Find the index of the first request within the interval
                idx = requests.bisect_left(interval_start)
                # Check if there are no requests within the interval
                if idx == len(requests) or requests[idx] > interval_end:
                    count += 1
            return count

        # Process each query
        result = []
        for query in queries:
            interval_start = query - x
            interval_end = query
            result.append(count_no_requests(interval_start, interval_end))

        return result