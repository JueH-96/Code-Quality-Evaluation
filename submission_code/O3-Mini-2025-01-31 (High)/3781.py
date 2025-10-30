import sys
sys.setrecursionlimit(10000)

class Solution:
    def maxDistance(self, side: int, points: list, k: int) -> int:
        """
        We are given a square of side length "side" with its boundary‐points (all points lie on one of the 4 edges)
        and a positive integer k. We want to choose k points among these points so that the minimum Manhattan
        distance over every pair is as large as possible.
        
        Our plan is to use binary search over the answer d (the candidate minimum Manhattan distance).
        For each candidate d we ask: "Is it possible to choose k points with pairwise Manhattan distance >= d?"
        Because k is small (<= 25), we use a DFS/backtracking search to test feasibility.
        
        Since the points lie on the boundary of the square, we first sort them according to their position
        along the perimeter. For that we compute a "perimeter coordinate" as follows:
          - Bottom edge (y==0): coordinate = x (from 0 to side).
          - Right edge (x==side): coordinate = side + y
          - Top edge (y==side): coordinate = 2*side + (side - x)
          - Left edge (x==0): coordinate = 3*side + (side - y)
        This ordering makes points that are very “close” on the boundary also come near in our sorted list.
        
        Then we binary search for the largest d in [1, 2*side] (note that with integer points, any two distinct points
        have Manhattan distance >= 1). For each candidate d we run a DFS which attempts (in the sorted order) to pick
        k points that are pairwise at least d apart.
        """
        
        # Helper to compute a coordinate along the square’s perimeter.
        def perim_coord(p):
            x, y = p
            if y == 0:
                return x
            elif x == side:
                return side + y
            elif y == side:
                return 2 * side + (side - x)
            else:  # x == 0
                return 3 * side + (side - y)
        
        # Convert given points to tuples and sort them by their perimeter order.
        pts = [(p[0], p[1]) for p in points]
        pts.sort(key=perim_coord)
        n = len(pts)
        
        # Helper: Manhattan distance between two points.
        def manhattan(a, b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        
        # For a candidate d, check if we can choose k points (from pts) such that 
        # every two chosen points have Manhattan distance at least d.
        def can_place(d):
            # For d <= 1, since coordinates are integers and points are distinct,
            # any two different points already satisfy a Manhattan distance >= 1.
            if d <= 1:
                return True
            
            found = False
            selected = []  # current list of chosen points
            
            # DFS: try to choose points starting from index "start" having "cnt" points already selected.
            def dfs(start, cnt):
                nonlocal found
                if found:
                    return
                if cnt == k:
                    found = True
                    return
                # Basic prune: not enough points remain.
                if n - start < (k - cnt):
                    return
                
                # Extra prune: count how many of the remaining pts are individually “compatible”
                # with the already selected ones.
                need = k - cnt
                valid_count = 0
                for j in range(start, n):
                    p = pts[j]
                    ok = True
                    for sp in selected:
                        if abs(p[0]-sp[0]) + abs(p[1]-sp[1]) < d:
                            ok = False
                            break
                    if ok:
                        valid_count += 1
                    if valid_count >= need:
                        break
                if valid_count < need:
                    return
                
                for j in range(start, n):
                    p = pts[j]
                    ok = True
                    for sp in selected:
                        if abs(p[0]-sp[0]) + abs(p[1]-sp[1]) < d:
                            ok = False
                            break
                    if not ok:
                        continue
                    selected.append(p)
                    dfs(j+1, cnt+1)
                    if found:
                        return
                    selected.pop()
            
            dfs(0, 0)
            return found
        
        # Binary search over candidate distances.
        # Lower bound is 1; upper bound is 2*side (maximum Manhattan distance between two corners).
        lo = 1
        hi = 2 * side + 1  # make hi an exclusive upper bound
        while lo < hi:
            mid = (lo + hi) // 2
            if can_place(mid):
                lo = mid + 1
            else:
                hi = mid
        return lo - 1  # lo-1 is the maximal d that worked.

        
# ---------------------------
# For testing (you can run several examples)
if __name__ == '__main__':
    sol = Solution()
    
    # Example 1:
    side1 = 2
    points1 = [[0,2],[2,0],[2,2],[0,0]]
    k1 = 4
    print(sol.maxDistance(side1, points1, k1))  # Expected output: 2

    # Example 2:
    side2 = 2
    points2 = [[0,0],[1,2],[2,0],[2,2],[2,1]]
    k2 = 4
    print(sol.maxDistance(side2, points2, k2))  # Expected output: 1
    
    # Example 3:
    side3 = 2
    points3 = [[0,0],[0,1],[0,2],[1,2],[2,0],[2,2],[2,1]]
    k3 = 5
    print(sol.maxDistance(side3, points3, k3))  # Expected output: 1