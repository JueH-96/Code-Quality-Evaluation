class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        """
        We want the length of the longest "increasing path" that MUST include
        coordinates[k]. An "increasing path" in 2D requires strictly increasing
        x and strictly increasing y. Concretely, if (x1, y1) -> (x2, y2) is
        part of the path, then x1 < x2 and y1 < y2.

        APPROACH:
        1) Let P = coordinates[k]. We will find:
              - dp1[i]: the length of the longest increasing path that ENDS at coordinates[i].
              - dp2[i]: the length of the longest increasing path that STARTS at coordinates[i].
           Then the answer for the point P is dp1[P] + dp2[P] - 1 (because P is counted twice).

        2) To compute dp1 (ends at i):
           - Sort all points by (x ascending, and if tie by y descending).
             The "tie by y descending" is a standard trick to ensure that if x_i == x_j,
             those points cannot be part of the same strictly-increasing chain.
           - Then, ignoring x (because it is already sorted), compute a 1D LIS over the y values
             (strictly increasing in y). We use a Fenwick tree (Binary Indexed Tree) for an
             O(n log n) LIS on the y dimension. dp1[i] is then the (LIS ending at i).

        3) To compute dp2 (starts at i):
           - We effectively reverse the x direction by mapping each point (x, y) -> (-x, -y).
             Then sort by these transformed coordinates with the same tie-breaking rule
             (ascending in -x, tie by descending in -y).
           - Perform the same 1D LIS (strictly increasing) in the transformed y' = -y dimension.
           - dp2[i] will be the LIS "ending" at the transformed point, which is the same as
             the LIS starting at (x, y) in the original space.

        4) Finally, for P = coordinates[k], the result is:
             dp1[P] + dp2[P] - 1

        5) Fenwick Tree (BIT) details for maximum-based query:
           - We compress y-coordinates so they fit in a smaller range [1..m].
           - fenwicksum[pos] will hold the maximum dp value for y <= pos.
           - To get dp for a new point with compressed y = cy, we do:
                dp[i] = 1 + fenwicksum.query(cy - 1)
             then update:
                fenwicksum.update(cy, dp[i])
           - This enforces strict y-increase, because we query (cy - 1).

        Time complexity: O(n log n), where n = len(coordinates).
        """

        import sys
        sys.setrecursionlimit(10**7)

        # Fenwick Tree helpers for "maximum" queries
        class FenwickTree:
            def __init__(self, size: int):
                self.size = size
                self.data = [0] * (size + 1)

            def update(self, idx: int, val: int):
                # Perform max-update while climbing up the Fenwick tree
                while idx <= self.size:
                    if val > self.data[idx]:
                        self.data[idx] = val
                    idx += idx & -idx

            def query(self, idx: int) -> int:
                # Get max from 1..idx
                result = 0
                while idx > 0:
                    if self.data[idx] > result:
                        result = self.data[idx]
                    idx -= idx & -idx
                return result

        # Step 1: identify the target point P
        px, py = coordinates[k]

        # We need a quick way to map from each (x,y) to its dp1 index and dp2 index.
        # Let's store (x, y, original_index).
        n = len(coordinates)
        points = [(x, y, i) for i, (x, y) in enumerate(coordinates)]

        # -------------------------------------------------------------------------
        # Step 2: Compute dp1
        # Sort by x ascending, if tie by y descending
        points.sort(key=lambda p: (p[0], -p[1]))
        # We'll need to do coordinate compression for the y dimension
        ys = [p[1] for p in points]
        unique_ys = sorted(set(ys))
        # map y -> compressed index
        compY = {}
        for idx, val in enumerate(unique_ys, start=1):
            compY[val] = idx

        dp1 = [0] * n  # dp1[i] = LIS ending at the point whose sorted index is i
        fenw = FenwickTree(len(unique_ys))

        for sorted_idx, (xv, yv, orig_i) in enumerate(points):
            cy = compY[yv]
            best_before = fenw.query(cy - 1)  # max dp among all < cy
            dp1_val = best_before + 1
            dp1[orig_i] = dp1_val
            fenw.update(cy, dp1_val)

        # -------------------------------------------------------------------------
        # Step 3: Compute dp2
        # We'll transform each point: (x, y) -> (-x, -y), then sort by (x' ascending, tie y' descending)
        # Then do the 1D LIS on y' (strictly increasing).
        points2 = [(-x, -y, i) for (x, y, i) in points]
        # Now sort by (x' ascending, tie by y' descending)
        points2.sort(key=lambda p: (p[0], -p[1]))

        # Coordinate compress the y' = -y
        ys2 = [p[1] for p in points2]
        unique_ys2 = sorted(set(ys2))
        compY2 = {}
        for idx, val in enumerate(unique_ys2, start=1):
            compY2[val] = idx

        dp2 = [0] * n
        fenw2 = FenwickTree(len(unique_ys2))

        for _, yv2, orig_i in points2:
            cy2 = compY2[yv2]
            best_before2 = fenw2.query(cy2 - 1)
            dp2_val = best_before2 + 1
            dp2[orig_i] = dp2_val
            fenw2.update(cy2, dp2_val)

        # -------------------------------------------------------------------------
        # Step 4: our target is the point P with original index = k
        # dp1[k] + dp2[k] - 1
        return dp1[k] + dp2[k] - 1