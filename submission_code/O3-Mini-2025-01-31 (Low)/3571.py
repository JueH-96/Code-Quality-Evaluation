from typing import List
import bisect

class BIT:
    def __init__(self, n):
        self.n = n
        self.data = [0]*(n+1)
        
    def update(self, i, val):
        # i: 1-indexed BIT update at position i with max operation
        while i <= self.n:
            if val > self.data[i]:
                self.data[i] = val
            i += i & -i
    
    def query(self, i):
        # query max in range [1,i]
        res = 0
        while i:
            if self.data[i] > res:
                res = self.data[i]
            i -= i & -i
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # Annotate points with original indices.
        points = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]
        
        # Coordinate compression for y
        ys = sorted({pt[1] for pt in points})
        # mapping: value -> index (1-indexed)
        y_to_idx = {y: i+1 for i, y in enumerate(ys)}
        size = len(ys)
        
        # dp1[i]: length of longest increasing chain ending at point i (in sorted order, increasing in x and y).
        dp1 = [0]*n  # we'll store corresponding to sorted order index
        # We will need to map original index -> dp1 value for the point in question.
        # Sort points ascending by (x, then y)
        points_asc = sorted(points, key=lambda p: (p[0], p[1]))
        
        bit = BIT(size)
        # We'll also store mapping from original index to dp1 value.
        orig_to_dp1 = {}
        for (x, y, orig) in points_asc:
            idx = y_to_idx[y]
            best = bit.query(idx - 1)  # strictly less y
            curr = best + 1
            dp1_val = curr
            bit.update(idx, curr)
            orig_to_dp1[orig] = dp1_val
        
        # dp2[i]: length of longest increasing chain starting at point i (if taken in increasing order).
        # We can compute this by reversing the order: consider points in descending order by x and y.
        dp2 = [0]*n
        orig_to_dp2 = {}
        points_desc = sorted(points, key=lambda p: (-p[0], -p[1]))
        bit = BIT(size)
        for (x, y, orig) in points_desc:
            idx = y_to_idx[y]
            # For dp2, we need to consider points with y strictly greater than current y.
            # In the BIT structure that works with "max query in prefix", it's easier to reverse the y order.
            # Instead, we can do a BIT that works on reversed y.
            # However, we can simulate similarly by querying on (size - idx) for reversed order.
            # But to keep it simple, we can instead modify BIT to query suffix maximum.
            # Instead, we can precompute reversed index.
            pass  # We haven't built BIT for suffix queries yet.
        
        # To compute dp2 quickly, we can transform y indices by mapping: ridx = size - idx + 1, so that "strictly greater y"
        # means: original y < some value in reversed order becomes: ridx < something. Let's do that.
        bit = BIT(size)
        orig_to_dp2 = {}
        for (x, y, orig) in points_desc:
            idx = y_to_idx[y]
            ridx = size - idx + 1  # reversed index, so that higher y gets lower ridx.
            # For strictly increasing y in original order, when going backwards,
            # we need to choose points that have y > current y. In reversed index, that means
            # ridx < (size - idx + 1)
            best = bit.query(ridx - 1)  # query maximum among indices < ridx in BIT using normal BIT (BIT used for prefix)
            curr = best + 1
            orig_to_dp2[orig] = curr
            # Now update BIT at ridx with value curr.
            bit.update(ridx, curr)
        
        # For the required point coordinates[k]:
        dp_start = orig_to_dp1[k]  # longest chain ending at P that includes P
        dp_end = orig_to_dp2[k]    # longest chain starting at P that includes P
        # Combining these, the longest chain that contains P is dp_start + dp_end - 1 (since point P is counted twice).
        return dp_start + dp_end - 1