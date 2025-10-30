class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        """
        We want a strictly-increasing chain: (x1,y1) < (x2,y2) < ... in both x and y.
        We must find the longest such chain that includes the point coordinates[k].
        The result is the length of that longest chain.

        Observing that this is equivalent to a 2D Longest Increasing Subsequence (LIS)
        problem (on x and y), we can do the following:

        1) Sort the points by x ascending; if x is the same, by y ascending.
           We'll assign each point an index in this sorted order. We'll keep track
           of where the original point k ended up in this sorted list.

        2) To compute LIS "forward" (dp_forward):
           - We'll process points in sorted order of x (and y).
           - For each x, we process all points that have this x but do not update
             the Fenwick tree until we've queried dp for them, preventing any usage
             of the same x for "strictly" increasing in x.
           - The Fenwick tree (or BIT) will store the maximum dp value for smaller y indices.
           - dp_forward[i] = 1 + max{dp_forward[j] | y_j < y_i, x_j < x_i}.

        3) To compute LIS "backward" (dp_backward):
           - Similarly, we'll sort points by x descending; if x is the same, by y descending.
           - Then do a similar Fenwick approach to find the LIS in reverse direction,
             effectively the chain from i to the "end".

        4) The answer for a particular point i is dp_forward[i] + dp_backward[i] - 1
           (because i is counted in both).

        We only need that for the index that corresponds to the original k point.

        We'll return dp_forward[idx_k] + dp_backward[idx_k] - 1.

        Complexity: O(n log n) due to sorting and Fenwick queries/updates.

        Fenwick tree details for "forward" pass:
          - We'll compress y to the range [1..n].
          - fenwicks[y] = max LIS ending with y' <= y.
          - dp_forward[i] = 1 + fenwicks.query(y_i - 1).
          Then after computing dp_forward[i] for all i in the same x-group,
          we update fenwicks.update(y_i, dp_forward[i]).

        We'll do a symmetrical approach for dp_backward.
        """

        import sys
        sys.setrecursionlimit(10**7)

        # Step 0: Setup
        n = len(coordinates)
        # Edge case: if n == 1, answer is obviously 1
        if n == 1:
            return 1

        # Keep track of the original index so we can identify which sorted index is k
        points = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]
        # Sort by x,y ascending for the forward pass
        points.sort(key=lambda x: (x[0], x[1]))

        # We'll need to find where the k-th original index ended up in this sorted list
        # for the forward pass. We'll store forward_idx_of[original_index] = sorted_index
        forward_idx_of = [0]*n
        for i, (_, _, original_idx) in enumerate(points):
            forward_idx_of[original_idx] = i
        k_idx_forward = forward_idx_of[k]

        # We'll also create a reversed-sorting array for backward pass
        points_reversed = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]
        # Sort by x descending, if tie, by y descending
        points_reversed.sort(key=lambda x: (-x[0], -x[1]))

        # We'll store backward_idx_of[original_index] = sorted_index in the reversed array
        backward_idx_of = [0]*n
        for i, (_, _, original_idx) in enumerate(points_reversed):
            backward_idx_of[original_idx] = i
        k_idx_backward = backward_idx_of[k]

        # Step 1: Coordinate compress y for forward direction (we only need the sorted y's)
        all_y = [p[1] for p in points]  # sorted in the same order as 'points'
        # but for compression we just need unique sorted
        unique_y = sorted(set(all_y))
        # map y -> compressed
        y_to_compressed = {}
        for i, val in enumerate(unique_y):
            y_to_compressed[val] = i+1  # 1-based index for Fenwick

        # Step 2: Fenwick Tree (BIT) for forward pass
        class Fenwick:
            def __init__(self, size):
                self.size = size
                self.tree = [0]*(size+1)

            def update(self, idx, val):
                while idx <= self.size:
                    self.tree[idx] = max(self.tree[idx], val)
                    idx += idx & -idx

            def query(self, idx):
                # max in range [1..idx]
                res = 0
                while idx > 0:
                    res = max(res, self.tree[idx])
                    idx -= idx & -idx
                return res

        fenwicks_forward = Fenwick(len(unique_y))
        dp_forward = [0]*n  # dp_forward[i] is LIS ending at sorted index i

        # We'll group points by x to avoid usage within the same x
        from collections import defaultdict
        groups_by_x = defaultdict(list)
        for i, (x, y, oidx) in enumerate(points):
            groups_by_x[x].append(i)

        # Now process groups in ascending order of x
        for x in sorted(groups_by_x.keys()):
            idxs = groups_by_x[x]
            # First compute dp for them using fenwicks.query
            tmp = []
            for i in idxs:
                _, y, _ = points[i]
                y_c = y_to_compressed[y]
                best_before = fenwicks_forward.query(y_c - 1)
                dp_forward[i] = best_before + 1
                tmp.append((y_c, dp_forward[i]))
            # Then update after
            for y_c, val in tmp:
                fenwicks_forward.update(y_c, val)

        # Step 3: do the same for backward pass, but we'll need to coordinate compress y
        # in the reversed-sorted array. We'll build dp_backward in terms of the reversed array.
        all_y_rev = [p[1] for p in points_reversed]
        unique_y_rev = sorted(set(all_y_rev))
        y_to_compressed_rev = {}
        for i, val in enumerate(unique_y_rev):
            y_to_compressed_rev[val] = i+1

        fenwicks_backward = Fenwick(len(unique_y_rev))
        dp_backward = [0]*n  # dp_backward[i] is LIS ending at sorted index i in reversed array

        groups_by_x_rev = defaultdict(list)
        # x is negative in sorting key, but let's just group by the actual x value
        for i, (x, y, oidx) in enumerate(points_reversed):
            groups_by_x_rev[x].append(i)

        for x in sorted(groups_by_x_rev.keys(), reverse=True):
            idxs = groups_by_x_rev[x]
            tmp = []
            for i in idxs:
                _, y, _ = points_reversed[i]
                y_c = y_to_compressed_rev[y]
                best_before = fenwicks_backward.query(y_c - 1)
                dp_backward[i] = best_before + 1
                tmp.append((y_c, dp_backward[i]))
            for y_c, val in tmp:
                fenwicks_backward.update(y_c, val)

        # Step 4: Combine results. We want dp_forward[k_idx] + dp_backward[k_idx] - 1,
        # BUT we must pick the correct indices. 
        # dp_forward is indexed by the forward-sorted array, so the relevant index for
        # the original k point is k_idx_forward.
        # dp_backward is indexed by the reversed-sorted array, so the relevant index for
        # the original k point is k_idx_backward.
        # The chain that includes the point k has length dp_forward[k_idx_forward] + dp_backward[k_idx_backward] - 1
        # because k is counted in both dp_forward and dp_backward.

        ans = dp_forward[k_idx_forward] + dp_backward[k_idx_backward] - 1
        return ans