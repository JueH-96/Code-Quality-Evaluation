import bisect

class SegmentTree:
    def __init__(self, data):
        self.n = len(data)
        self.size = 1
        while self.size < self.n:
            self.size <<= 1
        self.data = [0] * (2 * self.size)
        for i in range(self.n):
            self.data[self.size + i] = data[i]
        for i in range(self.n, self.size):
            self.data[self.size + i] = 0
        for i in range(self.size - 1, 0, -1):
            self.data[i] = max(self.data[2 * i], self.data[2 * i + 1])
    
    def update(self, pos, value):
        pos += self.size
        if self.data[pos] >= value:
            return
        self.data[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.data[2 * pos], self.data[2 * pos + 1])
            if self.data[pos] == new_val:
                break
            self.data[pos] = new_val
    
    def query(self, l, r):
        res = 0
        l += self.size
        r += self.size
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        points_with_idx = [(x, y, i) for i, (x, y) in enumerate(coordinates)]
        
        ys = [y for x, y in coordinates]
        unique_ys = sorted(set(ys))
        m = len(unique_ys)
        
        dp_ending = [1] * n
        data = [0] * m
        st_ending = SegmentTree(data)
        sorted_ending = sorted(points_with_idx, key=lambda p: (p[0], p[1]))
        for x, y, idx in sorted_ending:
            y_rank = bisect.bisect_left(unique_ys, y)
            max_val = 0
            if y_rank > 0:
                max_val = st_ending.query(0, y_rank - 1)
            dp_ending[idx] = max_val + 1
            current_val = st_ending.query(y_rank, y_rank)
            if dp_ending[idx] > current_val:
                st_ending.update(y_rank, dp_ending[idx])
        
        dp_starting = [1] * n
        data = [0] * m
        st_starting = SegmentTree(data)
        sorted_starting = sorted(points_with_idx, key=lambda p: (-p[0], -p[1]))
        for x, y, idx in sorted_starting:
            y_rank = bisect.bisect_left(unique_ys, y)
            max_val = 0
            if y_rank + 1 < m:
                max_val = st_starting.query(y_rank + 1, m - 1)
            dp_starting[idx] = max_val + 1
            current_val = st_starting.query(y_rank, y_rank)
            if dp_starting[idx] > current_val:
                st_starting.update(y_rank, dp_starting[idx])
        
        return dp_ending[k] + dp_starting[k] - 1