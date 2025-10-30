class Solution:
    def maxRemoval(self, nums: List[int], queries: List[List[int]]) -> int:
        """
        We need to find the largest subset of queries we can "discard" (remove)
        while still being able to zero out nums by choosing to decrement each
        index within the kept queries.  Equivalently, for each index i, the
        number of remaining queries covering i must be at least nums[i],
        because each query can contribute at most one decrement to that index.
        
        Let total_queries = len(queries).
        
        1) First check if it's even possible to reach zero using all queries:
           - For each i, let coveragePossible[i] = number of queries that cover i.
           - If coveragePossible[i] < nums[i] for any i, answer is -1 immediately.
           
        2) Otherwise, define 'removable[i] = coveragePossible[i] - nums[i]'. 
           This is how many queries covering i, at most, we can afford to remove
           without dropping below nums[i] coverage for that index.
           
        3) We then want to pick as many queries as possible to remove, subject to:
             For each i, the number of removed queries that cover i ≤ removable[i].
           In "difference-array" form, if we represent by r[] how many intervals
           we remove that include a position i, we must have r[i] ≤ removable[i]
           for each i.  Each removed query [l, r] adds +1 to r[x] for x in [l..r].
           
        4) A standard greedy to maximize the number of chosen (removed) intervals
           under per-point capacity constraints is:
             - Sort queries by their right endpoint ascending (if tie, by left ascending).
             - Keep a data structure (segment-tree with lazy propagation) that
               allows us to:
                 • Query the minimum value in [l..r].
                 • Apply a +1 (or -1 in a margin array) to all positions in [l..r].
               We check if we can remove the current query by verifying min in [l..r] ≥ 1
               in the "margin" array.  If yes, we remove it (apply -1).  Otherwise skip it.
           
           Concretely, we'll build an array margin[i] = removable[i].
           For each query (in order of ascending r), we check if
             min( margin[j] for j in [l..r] ) ≥ 1
           If so, we "remove" that query by subtracting 1 from margin[j] in that range.
           We count how many queries we successfully remove.
           
           This greedy is a known approach for maximum number of intervals subject
           to per-point capacities.  The result is the maximum removable queries.
        
        5) Implementation details:
           - We compute coveragePossible via a difference-array approach and prefix sum.
           - If any coveragePossible[i] < nums[i], return -1.
           - Otherwise, build margin[i] = coveragePossible[i] - nums[i].
           - Sort queries by r ascending, then process with a lazy-segment-tree
             that supports:
               • Range-minimum query
               • Range-add updates
           - Count how many queries we can remove. Return that count.
        
        The final answer is the number of queries we can remove.  If at the start
        it is impossible even with all queries, return -1.
        """
        
        import sys
        sys.setrecursionlimit(10**7)
        
        n = len(nums)
        m = len(queries)
        if n == 0 or m == 0:
            # If nums is empty or queries is empty, check trivial conditions.
            # If nums is empty => "already 0", can remove all queries => answer = m.
            # If queries is empty but nums not all 0 => -1 unless all nums are 0.
            return m if all(x == 0 for x in nums) else -1
        
        # Step 1) Compute coveragePossible for each index
        diff = [0]*(n+1)  # difference array of length n+1 to handle intervals
        for l,r in queries:
            diff[l] += 1
            if r+1 <= n-1:
                diff[r+1] -= 1
        
        coveragePossible = [0]*n
        coveragePossible[0] = diff[0]
        for i in range(1,n):
            coveragePossible[i] = coveragePossible[i-1] + diff[i]
        
        # Check feasibility with all queries
        for i in range(n):
            if coveragePossible[i] < nums[i]:
                return -1  # not feasible at all
        
        # Step 2) Build margin array = coveragePossible[i] - nums[i]
        margin = [coveragePossible[i] - nums[i] for i in range(n)]
        
        # Step 3) Sort queries by r ascending (if tie by l ascending)
        queries.sort(key=lambda x: (x[1], x[0]))
        
        # We'll implement a segment tree that supports:
        #   - range minimum query
        #   - range update (subtract 1)
        
        # Build segment tree over margin
        size = 1
        while size < n:
            size <<= 1
        
        INF = 10**15
        # We'll store the minimum in each segment
        segmin = [0]*(2*size)
        lazy = [0]*(2*size)   # for lazy propagation
        
        # Build function
        def build(arr):
            # put leaves at indices [size..size+n-1]
            for i in range(n):
                segmin[size + i] = arr[i]
            for i in range(n, size):
                segmin[size + i] = INF  # out-of-range
            # build internal nodes
            for i in range(size-1, 0, -1):
                segmin[i] = min(segmin[2*i], segmin[2*i+1])
        
        def apply_update(idx, val):
            segmin[idx] -= val
            lazy[idx] += val
        
        def push_down(idx):
            # push lazy values to children
            for c in (2*idx, 2*idx+1):
                apply_update(c, lazy[idx])
            lazy[idx] = 0
        
        def push_up(idx):
            segmin[idx] = min(segmin[2*idx], segmin[2*idx+1])
        
        # range update: subtract 'val' in [l..r]
        def update_range(l, r, val, idx, left, right):
            if r < left or right < l:
                return
            if l <= left and right <= r:
                apply_update(idx, val)
                return
            # partial overlap
            mid = (left + right)//2
            push_down(idx)
            update_range(l, r, val, 2*idx, left, mid)
            update_range(l, r, val, 2*idx+1, mid+1, right)
            push_up(idx)
        
        def query_min(l, r, idx, left, right):
            if r < left or right < l:
                return INF
            if l <= left and right <= r:
                return segmin[idx]
            mid = (left + right)//2
            push_down(idx)
            left_min = query_min(l, r, 2*idx, left, mid)
            right_min = query_min(l, r, 2*idx+1, mid+1, right)
            return min(left_min, right_min)
        
        # Build the tree
        build(margin)
        
        # Step 4) Greedily try to remove queries in sorted order
        removed_count = 0
        for l, r in queries:
            mn = query_min(l, r, 1, 0, size-1)
            if mn >= 1:
                # We can afford to remove this query
                removed_count += 1
                update_range(l, r, 1, 1, 0, size-1)  # subtract 1 in [l..r]
        
        return removed_count