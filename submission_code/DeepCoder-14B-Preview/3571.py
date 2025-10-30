import bisect

class SegmentTree:
    def __init__(self, size):
        self.n = 1
        while self.n < size:
            self.n <<= 1
        self.size = self.n
        self.tree = [0] * (2 * self.n)
    
    def update_point(self, pos, value):
        pos += self.n
        if self.tree[pos] >= value:
            return
        self.tree[pos] = value
        while pos > 1:
            pos >>= 1
            new_val = max(self.tree[2 * pos], self.tree[2 * pos + 1])
            if self.tree[pos] == new_val:
                break
            self.tree[pos] = new_val
    
    def query_range(self, l, r):
        res = 0
        l += self.n
        r += self.n
        while l < r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 1:
                r -= 1
                res = max(res, self.tree[r])
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maxPathLength(self, coordinates: list[list[int]], k: int) -> int:
        n = len(coordinates)
        if n == 0:
            return 0
        points = [(x, y, i) for i, (x, y) in enumerate(coordinates)]
        
        # Compute end[i]
        sorted_end = sorted(points, key=lambda p: (p[0], -p[1]))
        unique_ys = sorted({p[1] for p in points})
        m_end = len(unique_ys)
        end = [0] * n
        if m_end > 0:
            st_end = SegmentTree(m_end)
            for p in sorted_end:
                x, y, idx = p
                count = bisect.bisect_left(unique_ys, y)
                if count > 0:
                    max_val = st_end.query_range(0, count)
                else:
                    max_val = 0
                end[idx] = max_val + 1
                y_idx = bisect.bisect_left(unique_ys, y)
                st_end.update_point(y_idx, end[idx])
        
        # Compute start[i]
        sorted_start = sorted(points, key=lambda p: (-p[0], p[1]))
        m_start = m_end  # Using the same unique_ys since they are the same
        start = [0] * n
        if m_start > 0:
            st_start = SegmentTree(m_start)
            for p in sorted_start:
                x, y, idx = p
                count = bisect.bisect_right(unique_ys, y)
                if count < m_start:
                    max_val = st_start.query_range(count, m_start)
                else:
                    max_val = 0
                start[idx] = max_val + 1
                y_idx = bisect.bisect_left(unique_ys, y)
                st_start.update_point(y_idx, start[idx])
        
        # Calculate the result
        res = end[k] + start[k] - 1
        return res