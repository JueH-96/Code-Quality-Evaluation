import bisect

class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size_tree = 1
        while self.size_tree < self.n:
            self.size_tree <<= 1
        self.tree = [-float('inf')] * (2 * self.size_tree)
    
    def update(self, pos, value):
        idx = pos + self.size_tree
        if self.tree[idx] < value:
            self.tree[idx] = value
            idx >>= 1
            while idx >= 1:
                new_val = max(self.tree[2 * idx], self.tree[2 * idx + 1])
                if self.tree[idx] == new_val:
                    break
                self.tree[idx] = new_val
                idx >>= 1
    
    def query_range(self, l, r):
        res = -float('inf')
        l += self.size_tree
        r += self.size_tree
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.tree[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res

class Solution:
    def maximumSumQueries(self, nums1, nums2, queries):
        all_b = list(nums2)
        for x, y in queries:
            all_b.append(y)
        sorted_b = sorted(list(set(all_b)))
        m = len(sorted_b)
        
        points = sorted(
            zip(nums1, nums2, (a + b for a, b in zip(nums1, nums2))),
            key=lambda x: -x[0]
        )
        
        sorted_queries = []
        for idx, (x, y) in enumerate(queries):
            sorted_queries.append((-x, idx, x, y))
        sorted_queries.sort()
        sorted_queries = [(x, y, original_idx) for (neg_x, original_idx, x, y) in sorted_queries]
        
        ptr = 0
        result = [-1] * len(queries)
        st = SegmentTree(m)
        
        for x_i, y_i, original_idx in sorted_queries:
            while ptr < len(points) and points[ptr][0] >= x_i:
                a, b, s = points[ptr]
                idx_b = bisect.bisect_left(sorted_b, b)
                st.update(idx_b, s)
                ptr += 1
            l = bisect.bisect_left(sorted_b, y_i)
            if l >= m:
                res = -1
            else:
                max_val = st.query_range(l, m - 1)
                res = max_val if max_val != -float('inf') else -1
            result[original_idx] = res
        
        return result