class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        """
        We want the maximum length of an (x strictly increasing, y strictly increasing) path
        that must include coordinates[k]. We will find:
            prefix[i] = length of longest increasing path ending at point i
            suffix[i] = length of longest increasing path starting at point i
        Then the result for the point k is prefix[k] + suffix[k] - 1.

        The main challenge is efficiently computing these 'prefix' and 'suffix' values for
        all points in O(n log n). We do this by:
          1) Sort the points by (x ascending, then y ascending) to compute 'prefix'.
             We use a Fenwick (Binary Indexed) Tree to track the LIS in terms of y.
          2) Sort the points by (x descending, then y descending) to compute 'suffix'.
             Again use a Fenwick Tree for the LIS.

        The Fenwick Tree approach: When iterating over sorted points:
            - Compress the y-coordinates to make Fenwick indexing feasible.
            - For each point, dp = 1 + queryFenw( y-1 ), where queryFenw( y-1 ) 
              gives the maximum dp among all points with compressedY < current compressedY.
            - Update Fenw( y, dp ) = max of old vs dp.

        Finally, return prefix[k] + suffix[k] - 1.
        """

        # Helper Fenwick (Binary Indexed) Tree for max queries/updates
        class FenwickTree:
            def __init__(self, size):
                self.size = size
                self.data = [0]*(size+1)
            
            def update(self, idx, val):
                while idx <= self.size:
                    if val > self.data[idx]:
                        self.data[idx] = val
                    idx += idx & -idx

            def query(self, idx):
                res = 0
                while idx > 0:
                    if self.data[idx] > res:
                        res = self.data[idx]
                    idx -= idx & -idx
                return res

        n = len(coordinates)
        if n == 1:
            return 1  # only one point, so path length is 1

        # Separate list with indices
        indexed_coords = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]
        xk, yk = coordinates[k]

        # --------- Step 1: Compute prefix (longest inc path ending at each point) ---------

        # Sort points by x ascending, if tie then y ascending
        indexed_coords.sort(key=lambda x: (x[0], x[1]))
        # Coordinate-compress the y-values
        all_y = sorted({pt[1] for pt in indexed_coords})
        y_to_compressed = {}
        for i, val in enumerate(all_y, start=1):
            y_to_compressed[val] = i

        fenw = FenwickTree(len(all_y))
        prefix = [0]*n

        for xval, yval, idx in indexed_coords:
            cy = y_to_compressed[yval]
            best_before = fenw.query(cy - 1)  # max dp among smaller y
            prefix[idx] = best_before + 1
            fenw.update(cy, prefix[idx])

        # --------- Step 2: Compute suffix (longest inc path starting at each point) ---------

        # For suffix, we sort by x descending, if tie then y descending
        indexed_coords.sort(key=lambda x: (-x[0], -x[1]))
        # Rebuild Fenwicks
        fenw = FenwickTree(len(all_y))
        suffix = [0]*n

        for xval, yval, idx in indexed_coords:
            cy = y_to_compressed[yval]
            best_before = fenw.query(len(all_y) - cy)  # we can flip y to get strictly increasing in descending sort
            # Another approach is to re-map y so larger y become smaller index, then do the same logic
            # We'll do direct re-mapping: compressed index = len(all_y) + 1 - y_to_compressed[y]
            cy_reversed = len(all_y) + 1 - cy
            best_before = fenw.query(cy_reversed - 1)
            suffix[idx] = best_before + 1
            fenw.update(cy_reversed, suffix[idx])

        # The answer is prefix[k] + suffix[k] - 1
        return prefix[k] + suffix[k] - 1