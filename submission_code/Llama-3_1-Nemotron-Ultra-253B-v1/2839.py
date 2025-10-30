from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Sort the points in descending order of nums1 values
        points = sorted(zip(nums1, nums2), key=lambda x: (-x[0], -x[1]))
        
        # Collect all b values from points and y values from queries for coordinate compression
        all_b = []
        for a, b in points:
            all_b.append(b)
        for x, y in queries:
            all_b.append(y)
        all_b = sorted(list(set(all_b)))
        m = len(all_b)
        
        # Sort queries in descending order of x, keeping track of original indices
        sorted_queries = sorted([(queries[i], i) for i in range(len(queries))], key=lambda x: (-x[0][0], x[1]))
        
        # Initialize segment tree
        if m == 0:
            return [-1] * len(queries)
        st = self.SegmentTree(m)
        
        answers = [-1] * len(queries)
        i = 0  # Pointer to track inserted points
        
        for query, idx in sorted_queries:
            x, y = query
            # Insert all points with a >= x into the segment tree
            while i < len(points) and points[i][0] >= x:
                a, b = points[i]
                s = a + b
                compressed_b = bisect.bisect_left(all_b, b)
                if compressed_b < m and all_b[compressed_b] == b:
                    st.update(compressed_b, s)
                i += 1
            
            # Query the segment tree for the maximum sum where b >= y
            compressed_y = bisect.bisect_left(all_b, y)
            if compressed_y >= m:
                answers[idx] = -1
            else:
                max_s = st.query_range(compressed_y, m - 1)
                answers[idx] = max_s if max_s != -float('inf') else -1
        
        return answers
    
    class SegmentTree:
        def __init__(self, size):
            self.n = 1
            while self.n < size:
                self.n <<= 1
            self.tree = [-float('inf')] * (2 * self.n)
        
        def update(self, idx, value):
            idx += self.n
            if self.tree[idx] >= value:
                return
            self.tree[idx] = value
            while idx > 1:
                idx >>= 1
                new_val = max(self.tree[2 * idx], self.tree[2 * idx + 1])
                if self.tree[idx] == new_val:
                    break
                self.tree[idx] = new_val
        
        def query_range(self, l, r):
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