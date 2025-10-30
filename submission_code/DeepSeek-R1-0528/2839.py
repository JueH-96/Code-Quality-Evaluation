class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size = 1
        while self.size < self.n:
            self.size *= 2
        self.data = [-1] * (2 * self.size)
    
    def update(self, index, value):
        index += self.size
        if value > self.data[index]:
            self.data[index] = value
        while index > 1:
            index //= 2
            self.data[index] = max(self.data[2 * index], self.data[2 * index + 1])
    
    def query(self, l, r):
        l += self.size
        r += self.size
        res = -1
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l //= 2
            r //= 2
        return res

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        points = [(a, b) for a, b in zip(nums1, nums2)]
        b_vals = set()
        for a, b in points:
            b_vals.add(b)
        ys = set()
        for x, y in queries:
            ys.add(y)
        all_vals = b_vals | ys
        coords = sorted(all_vals)
        mapping = {val: idx for idx, val in enumerate(coords)}
        m = len(coords)
        
        points_sorted = sorted(points, key=lambda x: x[0], reverse=True)
        queries_sorted = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], key=lambda q: q[0], reverse=True)
        
        st = SegmentTree(m)
        j = 0
        n = len(points_sorted)
        ans = [-1] * len(queries)
        
        for x, y, idx in queries_sorted:
            while j < n and points_sorted[j][0] >= x:
                a, b = points_sorted[j]
                pos = mapping[b]
                st.update(pos, a + b)
                j += 1
            y_idx = mapping[y]
            res = st.query(y_idx, m - 1)
            ans[idx] = res
        
        return ans