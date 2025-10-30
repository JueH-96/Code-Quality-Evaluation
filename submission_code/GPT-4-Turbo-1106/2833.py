from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        from collections import defaultdict

        # Create a dictionary to store the last time each server received a request
        last_request_time = defaultdict(int)

        # Update the dictionary with the logs
        for server_id, time in logs:
            last_request_time[server_id] = max(last_request_time[server_id], time)

        # Initialize the result array
        result = []

        # Iterate through each query
        for query_time in queries:
            # Initialize the count of servers with no requests during the interval
            count = 0
            # Check each server
            for server_id in range(1, n + 1):
                # If the server did not receive any requests during the interval, increment the count
                if last_request_time[server_id] < query_time - x or last_request_time[server_id] > query_time:
                    count += 1
            # Append the count to the result array
            result.append(count)

        return result

# Example usage:
# sol = Solution()
# print(sol.countServers(3, [[1,3],[2,6],[1,5]], 5, [10,11]))  # Output: [1,2]
# print(sol.countServers(3, [[2,4],[2,1],[1,2],[3,1]], 2, [3,4]))  # Output: [0,1]