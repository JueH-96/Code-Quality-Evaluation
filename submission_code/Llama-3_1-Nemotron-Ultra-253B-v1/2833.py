from typing import List
from bisect import bisect_left, bisect_right

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.tree = [[] for _ in range(2 * self.size)]
        for i in range(self.n):
            self.tree[self.size + i] = [data[i]]
        for i in range(self.size - 1, 0, -1):
            left = self.tree[2 * i]
            right = self.tree[2 * i + 1]
            merged = []
            l = r = 0
            while l < len(left) or r < len(right):
                if l < len(left) and (r >= len(right) or left[l] <= right[r]):
                    merged.append(left[l])
                    l += 1
                else:
                    merged.append(right[r])
                    r += 1
            self.tree[i] = merged

    def query(self, l, r, x):
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res += len(self.tree[l]) - bisect_right(self.tree[l], x)
                l += 1
            if r % 2 == 0:
                res += len(self.tree[r]) - bisect_right(self.tree[r], x)
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def countServers(self, n: int, logs: List[List[int]], x: int, queries: List[int]) -> List[int]:
        logs.sort(key=lambda x: x[1])
        times = [log[1] for log in logs]
        servers = [log[0] for log in logs]
        m = len(logs)
        if m == 0:
            return [n] * len(queries)
        
        next_occurrence = [m] * m
        last_occurrence = dict()
        for i in range(m):
            server = servers[i]
            if server in last_occurrence:
                prev_i = last_occurrence[server]
                next_occurrence[prev_i] = i
            last_occurrence[server] = i
        
        st = SegmentTree(next_occurrence)
        
        res = []
        for q in queries:
            L = q - x
            R = q
            left = bisect_left(times, L)
            right_idx = bisect_right(times, R) - 1
            if left > right_idx:
                res.append(n)
            else:
                count = st.query(left, right_idx, right_idx)
                res.append(n - count)
        
        return res