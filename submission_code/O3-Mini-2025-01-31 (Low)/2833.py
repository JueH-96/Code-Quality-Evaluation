from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # Sort logs by time
        logs.sort(key=lambda log: log[1])
        
        # Prepare queries: pair each query with its index, and sort by query time.
        sorted_queries = sorted([(q, idx) for idx, q in enumerate(queries)])
        
        # Frequency array for the servers in current sliding window
        freq = [0] * (n + 1)  # servers are 1-indexed (given in problem)
        active = 0  # count of servers that have at least 1 log in current window
        
        ans = [0] * len(queries)
        left = 0
        right = 0
        m = len(logs)
        
        # Process each query in increasing time
        for qtime, q_idx in sorted_queries:
            # Add logs to the window: while log time <= current query time qtime
            while right < m and logs[right][1] <= qtime:
                server_id, time_val = logs[right]
                freq[server_id] += 1
                if freq[server_id] == 1:
                    active += 1
                right += 1
            # Remove logs that fall out of the window: log time < (qtime - x)
            while left < m and logs[left][1] < qtime - x:
                server_id, time_val = logs[left]
                freq[server_id] -= 1
                if freq[server_id] == 0:
                    active -= 1
                left += 1
            # The servers that did not get any request in [qtime - x, qtime]
            ans[q_idx] = n - active
        
        return ans

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1
    n = 3
    logs = [[1,3],[2,6],[1,5]]
    x = 5
    queries = [10,11]
    print(sol.countServers(n, logs, x, queries))  # Output: [1,2]
    
    # Example 2
    n = 3
    logs = [[2,4],[2,1],[1,2],[3,1]]
    x = 2
    queries = [3,4]
    print(sol.countServers(n, logs, x, queries))  # Output: [0,1]