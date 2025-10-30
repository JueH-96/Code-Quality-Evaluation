class Solution:
    def maximumSumQueries(self, nums1: List[int], nums2: List[int], queries: List[List[int]]) -> List[int]:
        import bisect

        n = len(nums1)
        # Combine (nums1, nums2) into a list of points
        points = [(nums1[i], nums2[i]) for i in range(n)]
        
        # We'll process points and queries in descending order of x.
        # For tie in x among points, we can keep them in descending order of y.
        points.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        # Collect all y-values (from points and queries) for coordinate compression
        all_y = []
        for x, y in points:
            all_y.append(y)
        for xq, yq in queries:
            all_y.append(yq)
        
        # Coordinate compression of y-values
        all_y = sorted(list(set(all_y)))
        
        # Helper to get compressed index
        def compress(val):
            # Find index in all_y of 'val' using bisect_left
            return bisect.bisect_left(all_y, val)
        
        # Fenwick (Binary Indexed) tree or Segment tree for range max queries
        # We'll store the maximum sum for each compressed y index
        size = len(all_y)
        fenwicks = [-1] * (size+1)  # 1-based indexing; store max sums
        
        # Fenwick tree update: set the max for position idx
        def fenwick_update(idx, value):
            # idx is 1-based in fenwicks
            while idx <= size:
                fenwicks[idx] = max(fenwicks[idx], value)
                idx += idx & -idx
        
        # Fenwick tree range query [idx..end] for maximum
        # We'll do a trick: we can only do prefix queries, so to get max from [idx..end],
        # we actually do max among all fenwicks from 1..size minus the prefix [1..(idx-1)] if we reversed.
        # Another simpler approach: manually do a downward query for max in [idx..size].
        def fenwick_range_max(start_idx):
            # Get maximum from start_idx..size
            # We'll do queries in two parts: max from [1..size] - max from [1..start_idx-1] if we had prefix sums
            # But Fenwick Tree is simpler to do prefix max. So let's do a direct approach for demonstration:
            # A direct approach is not typical for Fenwick's prefix design. Instead, we can build it to store
            # max in a prefix manner, but we need suffix queries. Let's store everything in reversed index.
            # For simplicity, let's do a segment tree approach. However, to keep it short, we'll do partial Fenwick approach as below.
            # We'll just do a linear downward from the top, but that is O(logN) for each step.
            
            # A standard approach: if we store y in descending order, then to get y >= start_idx, we want prefix up to start_idx in that reversed indexing. Let's do that quickly:
            # We'll store elements reversed so that bigger y has smaller index. We'll invert it. But let's keep the code as is for readability.
            # We'll do a manual "max from start_idx..end" by splitting into partial sums:
            
            # Because implementing a suffix max Fenwick quickly might be tricky in code golf. 
            # Instead let's do a small segment tree approach. But we'll do it carefully to keep the code short.

            # For efficiency, let's store everything in ascending y order; so "start_idx" means all indexes >= start_idx is from start_idx..(size-1).
            # We'll build Fenwick for normal prefix queries (1..idx).
            # Then max in [start_idx..size] = maximum among [1..size] minus ignoring the region [1..start_idx-1].
            # We can't "subtract" in a max Fenwick. We'll do a segment tree approach. 
            # For brevity, let's do a segment tree:

            return segtree_query(1, 0, size-1, start_idx, size-1)

        # Let's build a segment tree storing maximum sums
        segsize = 1
        while segsize < size:
            segsize *= 2
        segtree = [-1] * (2 * segsize)  # store max values
        
        def segtree_update(pos, val):
            # Update position pos with max(val, existing)
            pos += segsize
            segtree[pos] = max(segtree[pos], val)
            pos //= 2
            while pos > 0:
                segtree[pos] = max(segtree[2*pos], segtree[2*pos+1])
                pos //= 2
        
        def segtree_query(idx, left, right, ql, qr):
            # range max query in [ql, qr]
            if ql > right or qr < left:
                return -1
            if ql <= left and right <= qr:
                return segtree[idx]
            mid = (left + right) // 2
            return max(
                segtree_query(idx*2, left, mid, ql, qr),
                segtree_query(idx*2+1, mid+1, right, ql, qr),
            )
        
        # Sort queries by x descending to match points order
        # Keep track of original index to place answers properly
        sorted_queries = sorted([(x, y, i) for i, (x, y) in enumerate(queries)], key=lambda x: x[0], reverse=True)
        
        ans = [-1] * len(queries)
        pt_idx = 0  # index for points

        # Process each query from largest x to smallest x
        for xq, yq, qi in sorted_queries:
            # Move point pointer so that all points with x >= xq are processed
            while pt_idx < n and points[pt_idx][0] >= xq:
                px, py = points[pt_idx]
                cy = compress(py)
                segtree_update(cy, px + py)
                pt_idx += 1
            # Now get the best sum among all y >= yq
            cyq = compress(yq)
            best = segtree_query(1, 0, size-1, cyq, size-1)
            ans[qi] = best
        
        return ans