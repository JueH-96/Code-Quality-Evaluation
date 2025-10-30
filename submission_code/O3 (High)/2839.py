from bisect import bisect_left
from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # All (nums1, nums2) pairs together with their sum
        pairs = list(zip(nums1, nums2))
        pairs.sort(key=lambda p: -p[0])          # sort by nums1 in descending order
        
        # Prepare the queries with their original indices, sort by x in descending order
        q_data = [(idx, q[0], q[1]) for idx, q in enumerate(queries)]
        q_data.sort(key=lambda t: -t[1])         # sort by x descending
        
        # Coordinate-compress nums2 values
        all_b = sorted(set(nums2))
        m = len(all_b)
        
        # -------- segment tree for range maximum ----------
        size = 1
        while size < m:
            size <<= 1
        seg = [-1] * (2 * size)
        
        def seg_update(pos: int, val: int) -> None:
            """point update – keep maximum"""
            pos += size
            if val <= seg[pos]:
                return
            seg[pos] = val
            pos >>= 1
            while pos:
                new_val = max(seg[pos << 1], seg[(pos << 1) | 1])
                if new_val == seg[pos]:
                    break
                seg[pos] = new_val
                pos >>= 1
        
        def seg_query(l: int, r: int) -> int:
            """range maximum query on [l, r] (both inclusive)"""
            res = -1
            l += size
            r += size
            while l <= r:
                if l & 1:
                    res = max(res, seg[l])
                    l += 1
                if not (r & 1):
                    res = max(res, seg[r])
                    r -= 1
                l >>= 1
                r >>= 1
            return res
        # ---------------------------------------------------
        
        answers = [-1] * len(queries)
        ptr = 0  # pointer over pairs
        
        for idx, x, y in q_data:
            # insert all pairs whose nums1 >= x into the segment tree
            while ptr < n and pairs[ptr][0] >= x:
                a, b = pairs[ptr]
                s = a + b
                b_idx = bisect_left(all_b, b)
                seg_update(b_idx, s)
                ptr += 1
            
            # query the best sum with nums2 >= y
            b_idx = bisect_left(all_b, y)
            if b_idx < m:
                answers[idx] = seg_query(b_idx, m - 1)
            else:
                answers[idx] = -1
        
        return answers