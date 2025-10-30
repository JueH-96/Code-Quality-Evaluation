from typing import List
import bisect

# Fenwick Tree for maximum queries
class Fenw:
    def __init__(self, n):
        self.n = n
        self.tree = [0]*(n+1)
    
    def update(self, i, val):
        # i is 1-indexed
        while i <= self.n:
            if val > self.tree[i]:
                self.tree[i] = val
            i += i & -i
            
    def query(self, i):
        # returns maximum value in tree[1..i]
        res = 0
        while i:
            if self.tree[i] > res:
                res = self.tree[i]
            i -= i & -i
        return res

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # create a list of points: (x, y, original_index)
        points = []
        for i, (x,y) in enumerate(coordinates):
            points.append((x,y,i))
        
        # Coordinate compression for y values
        ys = sorted({y for x,y in points})
        # Make dictionary: y -> compressed value (1-indexed)
        comp = {y: idx+1 for idx, y in enumerate(ys)}
        M = len(ys)  # maximum index for y
        
        # For dp2 we need reverse order keys: let y_inv = M - comp[y] + 1
        # Build extended point info: (x, y, orig, yc, y_inv)
        pts = []
        for x, y, idx in points:
            yc = comp[y]
            y_inv = M - yc + 1
            pts.append((x, y, idx, yc, y_inv))
        
        # Sort the points in increasing order of x, then y.
        pts.sort(key=lambda item: (item[0], item[1]))
        
        # We'll compute dp1, longest chain ending at each point (chain: increasing in both x and y)
        dp1 = [0]*n
        fenw = Fenw(M)
        
        # Process points in groups that share the same x.
        i = 0
        while i < len(pts):
            j = i
            group = []
            while j < len(pts) and pts[j][0] == pts[i][0]:
                group.append(pts[j])
                j += 1
            # For each point in this group, compute dp1 = 1 + max_{q with q.y < current.y} dp1(q)
            temp_res = []
            for (x, y, orig, yc, y_inv) in group:
                best = fenw.query(yc - 1)  # Only those with y < current y
                curr_dp = best + 1
                dp1[orig] = curr_dp
                temp_res.append((yc, curr_dp))
            # After processing group, update fenw for all points in the group.
            for yc, curr_dp in temp_res:
                fenw.update(yc, curr_dp)
            i = j
        
        # Now compute dp2, longest chain starting at each point.
        # We want dp2[p] = 1 + max_{q: q.x > p.x and q.y > p.y} dp2[q].
        # We can process in reverse order (by increasing x) so that future points have been processed.
        dp2 = [0]*n
        fenw2 = Fenw(M)  # We will use the transformed coordinate y_inv
        # Process groups by x in descending order.
        i = len(pts) - 1
        while i >= 0:
            j = i
            group = []
            while j >= 0 and pts[j][0] == pts[i][0]:
                group.append(pts[j])
                j -= 1
            # For each point in the group, compute dp2 = 1 + max_{q with q.y > current.y} dp2(q)
            # Our mapping: q.y > current.y  <=>  comp[q.y] > comp[current.y]  <=>  y_inv(q) < y_inv(current)
            temp_res = []
            for (x, y, orig, yc, y_inv) in group:
                best = fenw2.query(y_inv - 1)  # Query for y_inv values strictly less than current's y_inv.
                curr_dp = best + 1
                dp2[orig] = curr_dp
                temp_res.append((y_inv, curr_dp))
            # After processing group, update fenw2.
            for y_inv, curr_dp in temp_res:
                fenw2.update(y_inv, curr_dp)
            i = j
        
        # The chain that includes point k can be constructed by taking
        # the best chain ending at coordinates[k] (dp1) and chain starting at coordinates[k] (dp2),
        # and subtracting 1 (because the point k counted twice).
        return dp1[k] + dp2[k] - 1


# Example usage:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    coordinates1 = [[3,1],[2,2],[4,1],[0,0],[5,3]]
    k1 = 1
    print(sol.maxPathLength(coordinates1, k1))  # Expected output: 3

    # Example 2:
    coordinates2 = [[2,1],[7,0],[5,6]]
    k2 = 2
    print(sol.maxPathLength(coordinates2, k2))  # Expected output: 2