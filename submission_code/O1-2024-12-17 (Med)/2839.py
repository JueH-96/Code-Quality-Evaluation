from typing import List
import bisect

class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums1)
        
        # Step 1: Create a list of (x, y, x+y) for all indices
        pairs = [(nums1[i], nums2[i], nums1[i] + nums2[i]) for i in range(n)]
        
        # Sort by descending x; if tie, by descending y
        pairs.sort(key=lambda p: (-p[0], -p[1]))
        
        # Step 2: Prepare queries in form (x_i, y_i, original_index), sorted by descending x_i
        # (If tie, by descending y_i to be consistent)
        Q = [(q[0], q[1], i) for i, q in enumerate(queries)]
        Q.sort(key=lambda x: (-x[0], -x[1]))
        
        # Step 3: Coordinate compression for the y-values (nums2 + query ys)
        all_y = set()
        for _, y, _ in Q:
            all_y.add(y)
        for _, y, _ in pairs:
            all_y.add(y)
        sorted_y = sorted(all_y)
        
        # Map actual y to compressed index
        comp = {}
        for i, val in enumerate(sorted_y):
            comp[val] = i
        
        # Segment Tree for range maximum on [0..len(sorted_y)-1]
        size = len(sorted_y)
        
        # We'll store -1 where there's no valid pair
        segtree = [-1] * (4 * size)
        
        # Build function (initially everything is -1, so we can skip an explicit build)
        
        # Update function: sets segtree[pos] = max(segtree[pos], val), 
        # then recurses up the tree
        def update(idx, start, end, pos, val):
            if start == end:
                segtree[idx] = max(segtree[idx], val)
                return
            mid = (start + end) // 2
            if pos <= mid:
                update(idx*2+1, start, mid, pos, val)
            else:
                update(idx*2+2, mid+1, end, pos, val)
            segtree[idx] = max(segtree[idx*2+1], segtree[idx*2+2])
        
        # Query function: returns max in [l..r]
        def query(idx, start, end, l, r):
            if r < start or end < l:
                return -1
            if l <= start and end <= r:
                return segtree[idx]
            mid = (start + end) // 2
            left_max = query(idx*2+1, start, mid, l, r)
            right_max = query(idx*2+2, mid+1, end, l, r)
            return max(left_max, right_max)
        
        # Step 4: Process queries in descending order of x
        # We'll keep a pointer over pairs, adding them to the segment tree as x >= query's x
        ans = [-1] * len(Q)
        p_index = 0
        for x_q, y_q, orig_idx in Q:
            # Add all pairs with x >= x_q
            while p_index < n and pairs[p_index][0] >= x_q:
                _, y_val, s = pairs[p_index]
                y_comp = comp[y_val]
                update(0, 0, size-1, y_comp, s)
                p_index += 1
            
            # Now query for all indices with y >= y_q
            # i.e. we want the range [comp[y_q]..end]
            if y_q in comp:
                y_start = comp[y_q]
                ans[orig_idx] = query(0, 0, size-1, y_start, size-1)
            else:
                # If y_q not in comp for any reason (shouldn't happen if we added all Y's),
                # we just get -1
                ans[orig_idx] = -1
        
        return ans