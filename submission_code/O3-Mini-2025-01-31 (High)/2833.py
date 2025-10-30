from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # First, sort the logs by time.
        logs.sort(key=lambda log: log[1])
        
        # Pair each query with its original index and sort by query time.
        queries_with_index = sorted([(q, i) for i, q in enumerate(queries)], key=lambda pair: pair[0])
        
        # Create a frequency array for servers (using 1-indexing).
        freq = [0] * (n + 1)
        distinct_count = 0  # Number of servers with at least one log in the current window.
        
        # Result array to store answers for each query.
        res = [0] * len(queries)
        
        left = 0  # Left pointer for maintaining the sliding window.
        right = 0  # Right pointer for adding new logs to the window.
        m = len(logs)
        
        # Process each query in increasing order of query time.
        for q_time, q_idx in queries_with_index:
            # Add logs up to the current query time.
            while right < m and logs[right][1] <= q_time:
                server_id, _ = logs[right]
                if freq[server_id] == 0:
                    distinct_count += 1
                freq[server_id] += 1
                right += 1
            
            # Remove logs that are outside the window [q_time - x, q_time].
            while left < m and logs[left][1] < q_time - x:
                server_id, _ = logs[left]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    distinct_count -= 1
                left += 1
            
            # The number of servers with no requests during [q_time - x, q_time].
            res[q_idx] = n - distinct_count
            
        return res

# Sample run to verify the code with provided examples.
if __name__ == "__main__":
    solution = Solution()

    # Example 1:
    n = 3
    logs = [[1,3],[2,6],[1,5]]
    x = 5
    queries = [10,11]
    print(solution.countServers(n, logs, x, queries))  # Expected output: [1,2]

    # Example 2:
    n = 3
    logs = [[2,4],[2,1],[1,2],[3,1]]
    x = 2
    queries = [3,4]
    print(solution.countServers(n, logs, x, queries))  # Expected output: [0,1]