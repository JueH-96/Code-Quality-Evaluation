import bisect

class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size_tree = 1
        while self.size_tree < self.n:
            self.size_tree <<= 1
        self.data = [-float('inf')] * (2 * self.size_tree)
    
    def update(self, pos, value):
        pos += self.size_tree
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
        res = -float('inf')
        l += self.size_tree
        r += self.size_tree
        while l <= r:
            if l % 2 == 1:
                res = max(res, self.data[l])
                l += 1
            if r % 2 == 0:
                res = max(res, self.data[r])
                r -= 1
            l >>= 1
            r >>= 1
        return res if res != -float('inf') else -1

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        events = []
        for a, b in zip(nums1, nums2):
            s = a + b
            events.append((-a, 0, b, s))
        for idx, (x, y) in enumerate(queries):
            events.append((-x, 1, y, idx))
        
        events.sort(key=lambda x: (x[0], x[1]))
        
        all_values = list(nums2)
        for q in queries:
            all_values.append(q[1])
        sorted_vals = sorted(list(set(all_values)))
        m = len(sorted_vals)
        if m == 0:
            return [-1] * len(queries)
        
        st = SegmentTree(m)
        res = [-1] * len(queries)
        
        for event in events:
            if event[1] == 0:
                _, _, b, s = event
                idx = bisect.bisect_left(sorted_vals, b)
                st.update(idx, s)
            else:
                _, _, y, q_idx = event
                idx = bisect.bisect_left(sorted_vals, y)
                if idx >= m:
                    res[q_idx] = -1
                else:
                    res[q_idx] = st.query(idx, m - 1)
        
        return res