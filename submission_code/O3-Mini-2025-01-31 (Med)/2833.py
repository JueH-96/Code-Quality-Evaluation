from typing import List

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        # First, sort the logs by timestamp.
        logs.sort(key=lambda entry: entry[1])
        
        # Next, sort the queries by query time while keeping track of their original indices.
        sorted_queries = sorted([(q, idx) for idx, q in enumerate(queries)], key=lambda x: x[0])
        
        # We create a frequency array for server requests (server ids are 1-indexed).
        freq = [0] * (n + 1)
        active_count = 0  # Number of servers with at least one request in the current window.
        answers = [0] * len(queries)
        
        left, right = 0, 0
        total_logs = len(logs)
        
        # Process each query in increasing order by time.
        for q_time, q_idx in sorted_queries:
            # Extend the window: include logs with time <= current query time.
            while right < total_logs and logs[right][1] <= q_time:
                server_id, time = logs[right]
                freq[server_id] += 1
                # If this is the first log in the window for this server, update active_count.
                if freq[server_id] == 1:
                    active_count += 1
                right += 1
            
            # Shrink the window from the left: remove logs with time < (q_time - x).
            window_lower_bound = q_time - x
            while left < right and logs[left][1] < window_lower_bound:
                server_id, time = logs[left]
                freq[server_id] -= 1
                # If after removal the server has no request in the window, update active_count.
                if freq[server_id] == 0:
                    active_count -= 1
                left += 1
            
            # The number of servers with no logs in [q_time - x, q_time] is (total servers - active servers).
            answers[q_idx] = n - active_count
        
        return answers

# The run() helper to process input and output.
def run():
    import sys
    data = sys.stdin.read().strip().split()
    if not data:
        return
    it = iter(data)
    
    # Read number of servers.
    n = int(next(it))
    # Next value is the number of logs.
    m = int(next(it))
    logs = []
    for _ in range(m):
        sid = int(next(it))
        t = int(next(it))
        logs.append([sid, t])
    
    # Read x.
    x = int(next(it))
    # Read number of queries.
    q_len = int(next(it))
    queries = [int(next(it)) for _ in range(q_len)]
    
    sol = Solution()
    result = sol.countServers(n, logs, x, queries)
    sys.stdout.write(" ".join(map(str, result)))

if __name__ == '__main__':
    run()