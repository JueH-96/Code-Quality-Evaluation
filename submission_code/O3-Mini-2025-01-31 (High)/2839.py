from typing import List
from bisect import bisect_left

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        # Build candidate list: (nums1 value, nums2 value, sum)
        candidates = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        # Sort candidates in descending order by nums1 value.
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        # Prepare queries with original indices.
        # Each query: [x, y] from queries.
        qlist = [(q[0], q[1], i) for i, q in enumerate(queries)]
        # Sort queries descending by x.
        qlist.sort(key=lambda x: x[0], reverse=True)
        
        # We will coordinate compress based on nums2 from candidates and y from queries.
        comp_vals = set()
        for a, b, _ in candidates:
            comp_vals.add(b)
        for _, y, _ in qlist:
            comp_vals.add(y)
        sorted_vals = sorted(comp_vals)
        m = len(sorted_vals)
        
        # BIT: We'll implement a Fenwick Tree for range maximum query.
        # We'll use 1-indexing. The BIT indices will be based on reversed order of sorted_vals.
        # Mapping function: For a given value v, its BIT index is: idx = m - bisect_left(sorted_vals, v)
        BIT = [-1] * (m + 1)
        
        def update(idx: int, val: int) -> None:
            # BIT update: update idx and all appropriate indices.
            while idx <= m:
                BIT[idx] = max(BIT[idx], val)
                idx += idx & -idx
        
        def query(idx: int) -> int:
            # BIT query: maximum in prefix [1, idx]
            res = -1
            while idx:
                res = max(res, BIT[idx])
                idx -= idx & -idx
            return res
        
        # Process queries:
        res = [-1] * len(queries)
        cand_idx = 0
        for x_val, y_val, q_idx in qlist:
            # Add all candidates that satisfy nums1 >= x_val.
            while cand_idx < len(candidates) and candidates[cand_idx][0] >= x_val:
                _, cand_y, cand_sum = candidates[cand_idx]
                # Get BIT index using reversed ordering.
                pos = bisect_left(sorted_vals, cand_y)
                bit_index = m - pos  # candidate's BIT index
                update(bit_index, cand_sum)
                cand_idx += 1
            # Answer the query: we need candidate with nums2 >= y_val.
            pos = bisect_left(sorted_vals, y_val)
            if pos == len(sorted_vals):
                # No candidate with nums2 >= y_val exists
                res[q_idx] = -1
            else:
                bit_index = m - pos  # BIT query on prefix [1, bit_index] gives max among candidates with nums2 >= y_val.
                res[q_idx] = query(bit_index)
        return res