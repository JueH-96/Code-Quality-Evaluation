from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        # Prepare points (a, b, sum)
        pts = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        # Collect all ys for compression: nums2 values and query y's
        ys = set(nums2)
        for x, y in queries:
            ys.add(y)
        sorted_ys = sorted(ys)
        # Map y to compressed index
        y_to_idx = {y: i for i, y in enumerate(sorted_ys)}
        m = len(sorted_ys)
        
        # Build a segment tree for range-max queries over y-indices [0..m-1]
        size = 1
        while size < m:
            size <<= 1
        seg = [-1] * (2 * size)
        
        # Helper to update position i (0-based) with value v (take max)
        def seg_update(i: int, v: int):
            i += size
            if v <= seg[i]:
                return
            seg[i] = v
            i //= 2
            while i:
                newv = seg[2*i] if seg[2*i] >= seg[2*i+1] else seg[2*i+1]
                if seg[i] == newv:
                    break
                seg[i] = newv
                i //= 2
        
        # Helper to query max on [l..r] inclusive
        def seg_query(l: int, r: int) -> int:
            res = -1
            l += size
            r += size
            while l <= r:
                if (l & 1) == 1:
                    if seg[l] > res:
                        res = seg[l]
                    l += 1
                if (r & 1) == 0:
                    if seg[r] > res:
                        res = seg[r]
                    r -= 1
                l //= 2
                r //= 2
            return res
        
        # Sort points by nums1 descending
        pts.sort(key=lambda x: x[0], reverse=True)
        # Prepare queries with original indices, sort by x descending
        q_with_idx = [(x, y, i) for i, (x, y) in enumerate(queries)]
        q_with_idx.sort(key=lambda x: x[0], reverse=True)
        
        ans = [-1] * len(queries)
        pi = 0  # pointer in pts
        # Process each query in descending x order
        for x, y, qi in q_with_idx:
            # Insert all points with a >= x into the segment tree
            while pi < n and pts[pi][0] >= x:
                _, b, s = pts[pi]
                bi = y_to_idx[b]
                seg_update(bi, s)
                pi += 1
            # Now query the max sum for all b >= y
            yi = y_to_idx[y]
            res = seg_query(yi, m - 1)
            ans[qi] = res
        
        return ans