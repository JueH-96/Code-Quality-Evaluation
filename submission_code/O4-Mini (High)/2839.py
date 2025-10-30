from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        # Pair up nums1 and nums2 and sort by nums1 descending
        items = list(zip(nums1, nums2))
        items.sort(key=lambda x: x[0], reverse=True)

        # Coordinate-compress nums2 values
        ys = sorted(set(nums2))
        M = len(ys)

        # Fenwick (BIT) for range-maximum on prefixes
        bit = [0] * (M + 1)
        def bit_update(i: int, val: int):
            # point-update: set bit[i] = max(bit[i], val)
            while i <= M:
                if bit[i] < val:
                    bit[i] = val
                i += i & -i

        def bit_query(i: int) -> int:
            # prefix-max query on [1..i]
            res = 0
            while i > 0:
                if bit[i] > res:
                    res = bit[i]
                i -= i & -i
            return res

        # Prepare queries: (x, y, original_index), sorted by x descending
        q_list = [(x, y, idx) for idx, (x, y) in enumerate(queries)]
        q_list.sort(key=lambda t: t[0], reverse=True)

        ans = [-1] * len(queries)
        ptr = 0  # pointer into items

        # Process each query in descending order of x
        for x, y, qidx in q_list:
            # Insert all items with nums1 >= x into the BIT
            while ptr < n and items[ptr][0] >= x:
                a, b = items[ptr]
                s = a + b
                # find compressed position of b in ys
                p = bisect.bisect_left(ys, b)
                # we map p -> reversed index so that suffix-queries become prefix-queries
                rev = M - 1 - p
                bit_update(rev + 1, s)
                ptr += 1

            # Now answer the query: we need max sum for nums2 >= y
            p_y = bisect.bisect_left(ys, y)
            if p_y == M:
                # no nums2 value >= y
                continue
            rev_y = M - 1 - p_y
            best = bit_query(rev_y + 1)
            if best > 0:
                ans[qidx] = best
            # if best == 0, it means no valid insertion, ans[qidx] stays -1

        return ans