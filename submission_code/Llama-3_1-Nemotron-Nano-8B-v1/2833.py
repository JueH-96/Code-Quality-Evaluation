from typing import List

class FenwickTree:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 2)  # 1-based indexing

    def update(self, idx, delta):
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def query(self, idx):
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_query(self, l, r):
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda t: t[1])  # Sort logs by time
        sorted_queries = sorted(enumerate(queries), key=lambda q: q[1][1])  # Sort queries by their end time
        max_time = 10**6
        ft = FenwickTree(max_time)
        last_time = [0] * n  # Track the latest time for each server
        res = [0] * len(queries)
        i = 0  # Pointer for logs

        for q in sorted_queries:
            original_idx, a = q[0], q[1][0]
            b = q[1][1]
            # Process all logs with time <= current b
            while i < len(logs) and logs[i][1] <= b:
                server = logs[i][0] - 1  # Convert to 0-based index
                old = last_time[server]
                if old > 0:
                    ft.update(old, -1)
                new = max(old, logs[i][1])
                last_time[server] = new
                ft.update(new, 1)
                i += 1
            # Calculate the sum of servers active in [a, b]
            sum_active = ft.range_query(a, b)
            res[original_idx] = n - sum_active  # Number of inactive servers

        return res