class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        """
        We want the length of the longest strictly increasing path (x and y both strictly increasing)
        that includes coordinates[k].

        Key idea:
        1) Consider dp_end[i] = length of the longest chain ending at point i.
        2) Consider dp_start[i] = length of the longest chain starting at point i.
        3) Then the answer for point i is dp_end[i] + dp_start[i] - 1 (we subtract 1 so that i
           is not double-counted).
        4) We only need dp_end[] and dp_start[] for all points, then the result is dp_end[k] + dp_start[k] - 1.

        Computing dp_end[]:
        - Sort points by ascending x, and break ties by ascending y.
          Group points with the same x so that no transitions happen between points of the same x (strict x < x).
        - Use a Fenwick (BIT) for the y-coordinates (after coordinate-compressing them).
        - dp_end for a point with y-compressed index cy is: 1 + maxFenwicksum(cy - 1).
          Then update Fenwicksum at cy with dp_end.

        Computing dp_start[]:
        - We do the same idea but effectively from right to left.
        - An easy trick is to negate x and y: new point = (-x, -y).  A strictly increasing chain in (x,y)
          becomes a strictly increasing chain in (-x,-y) if we sort those in ascending order of (-x).
        - Then dp_start[i] is dp_end'[i] if we compute dp_end over the negated points.

        This yields an O(n log n) solution (for sorting + Fenwicksum).
        """

        import sys
        import bisect
        from itertools import groupby

        n = len(coordinates)
        # Build an array of (x, y, original_index)
        points = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]

        # 1) Compute dp_end via Fenwicksum on y after sorting by x ascending, then y ascending

        # Coordinate-compress y
        all_y = sorted({p[1] for p in points})
        y2c = {}
        for i_c, val in enumerate(all_y):
            y2c[val] = i_c + 1  # 1-based index

        # Sort by x ascending, then y ascending
        points.sort(key=lambda p: (p[0], p[1]))

        # Fenwicksum for maximum
        M = len(all_y)
        fenwicksum = [0] * (M + 1)

        def fenwicksum_query(idx):
            """Returns the max in Fenwicksum from 1..idx."""
            ans = 0
            while idx > 0:
                ans = max(ans, fenwicksum[idx])
                idx -= idx & -idx
            return ans

        def fenwicksum_update(idx, val):
            """Updates Fenwicksum so that fenwicksum[idx] = max(fenwicksum[idx], val)."""
            while idx <= M:
                fenwicksum[idx] = max(fenwicksum[idx], val)
                idx += idx & -idx

        dp_end = [0] * n

        # Group points by their x-value to ensure we don't transition between points of the same x
        for xval, group in groupby(points, key=lambda p: p[0]):
            batch = list(group)
            # First compute dp_end for each in this batch
            temp_store = []
            for (x, y, i_orig) in batch:
                cy = y2c[y]
                best_prev = fenwicksum_query(cy - 1)
                dp_end[i_orig] = best_prev + 1
                temp_store.append((cy, dp_end[i_orig]))
            # Now update the fenwicksum with these dp values
            for (cy, val) in temp_store:
                fenwicksum_update(cy, val)

        # 2) Compute dp_start by the same logic, but on points negated
        points_neg = [(-p[0], -p[1], p[2]) for p in points]
        # Compress the negated y
        all_y_neg = sorted({p[1] for p in points_neg})
        y2c_neg = {}
        for i_c, val in enumerate(all_y_neg):
            y2c_neg[val] = i_c + 1

        # Sort by x' ascending (which is -x ascending => x descending), tie by y' ascending
        points_neg.sort(key=lambda p: (p[0], p[1]))

        M2 = len(all_y_neg)
        fenwicksum2 = [0] * (M2 + 1)

        def fenwicksum_query2(idx):
            ans = 0
            while idx > 0:
                ans = max(ans, fenwicksum2[idx])
                idx -= idx & -idx
            return ans

        def fenwicksum_update2(idx, val):
            while idx <= M2:
                fenwicksum2[idx] = max(fenwicksum2[idx], val)
                idx += idx & -idx

        dp_start = [0] * n

        for xval, group in groupby(points_neg, key=lambda p: p[0]):
            batch = list(group)
            temp_store = []
            for (x2, y2, i_orig) in batch:
                cy2 = y2c_neg[y2]
                best_prev = fenwicksum_query2(cy2 - 1)
                dp_start[i_orig] = best_prev + 1
                temp_store.append((cy2, dp_start[i_orig]))
            for (cy2, val) in temp_store:
                fenwicksum_update2(cy2, val)

        # Finally, the answer is dp_end[k] + dp_start[k] - 1
        return dp_end[k] + dp_start[k] - 1