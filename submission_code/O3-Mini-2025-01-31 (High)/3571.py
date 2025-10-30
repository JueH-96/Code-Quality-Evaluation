from typing import List

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        n = len(coordinates)
        # Create a list of points along with their original index.
        points = []
        for i, (x, y) in enumerate(coordinates):
            points.append((x, y, i))
        
        # Coordinate compression on y-values.
        unique_ys = sorted({y for _, y, _ in points})
        # Map each y to a rank (1-indexed).
        y_to_rank = {y: i+1 for i, y in enumerate(unique_ys)}
        size = len(unique_ys)  # number of unique y values.
        
        # dp1[i] will store the length of the longest increasing chain ending at point i.
        dp1 = [0] * n
        
        # Fenwick tree (Binary Indexed Tree) for maximum queries (for dp1).
        fenw1 = [0] * (size + 1)  # Index 1..size
        
        def fenw_query(idx: int) -> int:
            res = 0
            while idx:
                if fenw1[idx] > res:
                    res = fenw1[idx]
                idx -= idx & -idx
            return res
        
        def fenw_update(idx: int, val: int):
            while idx <= size:
                if val > fenw1[idx]:
                    fenw1[idx] = val
                idx += idx & -idx
        
        # Process points in increasing order of x (and y).
        points.sort(key=lambda p: (p[0], p[1]))
        i = 0
        while i < len(points):
            j = i
            group = []
            current_x = points[i][0]
            # Gather all points with the same x.
            while j < len(points) and points[j][0] == current_x:
                group.append(points[j])
                j += 1
            # For all points in this group, compute dp1 without updating Fenw tree immediately.
            for x, y, orig_idx in group:
                rank_y = y_to_rank[y]
                best_before = fenw_query(rank_y - 1)  # Only consider points with y < current y.
                dp1[orig_idx] = best_before + 1
            # Now update the Fenw tree with the new dp1 values.
            for x, y, orig_idx in group:
                rank_y = y_to_rank[y]
                fenw_update(rank_y, dp1[orig_idx])
            i = j
        
        # dp2[i] will store the length of the longest increasing chain starting at point i.
        dp2 = [0] * n
        # We'll use a second Fenw tree for dp2.
        fenw2 = [0] * (size + 1)
        
        # For dp2, we want to extend from a point p to a point q if (p.x, p.y) < (q.x, q.y).
        # We'll process points in descending order of x and descending order of y.
        # However, we need to query based on y in the "upper" side.
        # We transform the compressed y value: new_y = size - y_rank + 1.
        # Then, for a point p with new_y, any point q with q.y > p.y will have new_y_q < new_y.
        def fenw2_query(idx: int) -> int:
            res = 0
            while idx:
                if fenw2[idx] > res:
                    res = fenw2[idx]
                idx -= idx & -idx
            return res
        
        def fenw2_update(idx: int, val: int):
            while idx <= size:
                if val > fenw2[idx]:
                    fenw2[idx] = val
                idx += idx & -idx
        
        # Sort points in descending order (by x then y).
        points.sort(key=lambda p: (p[0], p[1]), reverse=True)
        i = 0
        while i < len(points):
            j = i
            group = []
            current_x = points[i][0]
            while j < len(points) and points[j][0] == current_x:
                group.append(points[j])
                j += 1
            for x, y, orig_idx in group:
                rank_y = y_to_rank[y]
                new_y = size - rank_y + 1
                # Query the fenw2 for indices strictly less than new_y;
                # these correspond to points with original y greater than current y.
                best_after = fenw2_query(new_y - 1) if new_y > 1 else 0
                dp2[orig_idx] = best_after + 1
            for x, y, orig_idx in group:
                rank_y = y_to_rank[y]
                new_y = size - rank_y + 1
                fenw2_update(new_y, dp2[orig_idx])
            i = j
        
        # The longest increasing path that contains the point coordinates[k]
        # can be obtained by joining the best chain ending at that point (dp1)
        # with the best chain starting at that point (dp2), subtracting 1 for the overlap.
        return dp1[k] + dp2[k] - 1