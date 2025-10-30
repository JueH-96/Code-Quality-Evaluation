import bisect
from typing import List

class SegmentTree:
    def __init__(self, size):
        self.n = size
        self.size_tree = 1
        while self.size_tree < self.n:
            self.size_tree <<= 1
        self.tree = [-float('inf')] * (2 * self.size_tree)

    def update_point(self, pos, value):
        pos += self.size_tree
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
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        elements = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        elements.sort(key=lambda x: (-x[0], -x[1]))
        
        # Collect all nums2 values and y's from queries
        all_b = []
        for a, b, s in elements:
            all_b.append(b)
        for x, y in queries:
            all_b.append(y)
        # Deduplicate and sort
        sorted_b = sorted(list(set(all_b)))
        m = len(sorted_b)
        
        # Prepare queries with their original indices
        q_with_indices = [(query[0], query[1], idx) for idx, query in enumerate(queries)]
        q_with_indices.sort(key=lambda x: (-x[0]))
        
        # Initialize the segment tree
        seg = SegmentTree(m)
        ptr = 0
        res = [-1] * len(queries)
        
        for x, y, idx in q_with_indices:
            # Add all elements with nums1 >= x
            while ptr < n and elements[ptr][0] >= x:
                a_elem, b_elem, s_elem = elements[ptr]
                # Find rank of b_elem in sorted_b
                b_pos = bisect.bisect_left(sorted_b, b_elem)
                if b_pos < len(sorted_b) and sorted_b[b_pos] == b_elem:
                    seg.update_point(b_pos, s_elem)
                ptr += 1
            
            # Query for y
            y_pos = bisect.bisect_left(sorted_b, y)
            if y_pos >= m:
                res[idx] = -1
            else:
                max_sum = seg.query_range(y_pos, m - 1)
                if max_sum == -float('inf'):
                    res[idx] = -1
                else:
                    res[idx] = max_sum
        
        return res