from bisect import bisect_left
from typing import List

class SegmentTree:
    def __init__(self, data_size):
        self.n = 1
        while self.n < data_size:
            self.n <<= 1
        self.size = self.n * 2
        self.tree = [-float('inf')] * (self.size)
    
    def update(self, index, value):
        if index >= self.n:
            return
        index += self.n
        if self.tree[index] < value:
            self.tree[index] = value
            index >>= 1
            while index >= 1:
                new_val = max(self.tree[index * 2], self.tree[index * 2 + 1])
                if self.tree[index] == new_val:
                    break
                self.tree[index] = new_val
                index >>= 1
    
    def query_range_max(self, l, r):
        res = -float('inf')
        l += self.n
        r += self.n
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
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        all_b = nums2.copy()
        for x, y in queries:
            all_b.append(y)
        all_b_sorted = sorted(list(set(all_b)))
        elements = [(a, b, a + b) for a, b in zip(nums1, nums2)]
        elements.sort(key=lambda x: (-x[0], -x[1]))
        sorted_queries = sorted([(x, y, idx) for idx, (x, y) in enumerate(queries)], key=lambda q: (-q[0], -q[1]))
        m = len(all_b_sorted)
        if m == 0:
            return [-1] * len(queries)
        st = SegmentTree(m)
        result = [-1] * len(queries)
        ptr = 0
        for x, y, original_idx in sorted_queries:
            while ptr < len(elements) and elements[ptr][0] >= x:
                a_elem, b_elem, sum_elem = elements[ptr]
                idx = bisect_left(all_b_sorted, b_elem)
                st.update(idx, sum_elem)
                ptr += 1
            idx_y = bisect_left(all_b_sorted, y)
            if idx_y >= m:
                result[original_idx] = -1
            else:
                max_sum = st.query_range_max(idx_y, m - 1)
                result[original_idx] = max_sum if max_sum != -float('inf') else -1
        return result