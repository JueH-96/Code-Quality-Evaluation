from typing import List

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        # Offline approach: sort points by nums1 desc, queries by x desc.
        n = len(nums1)
        q = len(queries)
        # Collect all nums2 values and query y's for compression
        coords = set(nums2)
        for _, y in queries:
            coords.add(y)
        sorted_coords = sorted(coords)
        # map original nums2 or y to compressed index 0..m-1
        comp = {v: i for i, v in enumerate(sorted_coords)}
        m = len(sorted_coords)
        
        # Prepare points: (nums1, nums2, sum)
        points = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        points.sort(key=lambda x: -x[0])  # descending by nums1
        
        # Prepare queries: add original index, sort by x desc
        queries_with_idx = [(x, y, idx) for idx, (x, y) in enumerate(queries)]
        queries_with_idx.sort(key=lambda x: -x[0])
        
        # Segment tree for range max over compressed nums2 index
        size = 1
        while size < m:
            size <<= 1
        seg = [-10**30] * (2 * size)
        
        def seg_update(pos: int, val: int):
            # set seg[pos] = max(existing, val)
            i = pos + size
            seg[i] = max(seg[i], val)
            i //= 2
            while i:
                seg[i] = max(seg[2*i], seg[2*i+1])
                i //= 2
        
        def seg_query(l: int, r: int) -> int:
            # max on interval [l, r)
            res = -10**30
            l += size
            r += size
            while l < r:
                if l & 1:
                    res = max(res, seg[l])
                    l += 1
                if r & 1:
                    r -= 1
                    res = max(res, seg[r])
                l //= 2
                r //= 2
            return res
        
        ans = [-1] * q
        pi = 0  # pointer in points
        # Process each query
        for x, y, qi in queries_with_idx:
            # Insert all points with nums1 >= x into seg tree
            while pi < n and points[pi][0] >= x:
                _, yj, s = points[pi]
                seg_update(comp[yj], s)
                pi += 1
            # Query max sum among points with nums2 >= y
            # compressed y index
            cy = comp.get(y, None)
            if cy is None:
                # no such y in compression => no point satisfies
                ans[qi] = -1
            else:
                res = seg_query(cy, m)
                ans[qi] = res if res > -10**29 else -1
        
        return ans