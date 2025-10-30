import bisect
from typing import List

class Solution:
    def maxDistance(self, side: int, points: List[List[int]], k: int) -> int:
        
        # Segment Tree for Range Max Query.
        # It is defined inside the main method to be re-initialized for each `can_place` call.
        class SegmentTree:
            def __init__(self, size):
                self.size = size
                if size == 0:
                    self.tree = []
                else:
                    self.tree = [0] * (4 * size)

            def _update(self, node, start, end, idx, val):
                if start == end:
                    self.tree[node] = max(self.tree[node], val)
                    return
                mid = (start + end) // 2
                if start <= idx <= mid:
                    self._update(2 * node, start, mid, idx, val)
                else:
                    self._update(2 * node + 1, mid + 1, end, idx, val)
                self.tree[node] = max(self.tree[2 * node], self.tree[2 * node + 1])
            
            def update(self, idx, val):
                if not self.tree: return
                self._update(1, 0, self.size - 1, idx, val)

            def _query(self, node, start, end, l, r):
                if r < start or end < l or l > r:
                    return 0
                if l <= start and end <= r:
                    return self.tree[node]
                mid = (start + end) // 2
                p1 = self._query(2 * node, start, mid, l, r)
                p2 = self._query(2 * node + 1, mid + 1, end, l, r)
                return max(p1, p2)

            def query(self, l, r):
                if not self.tree or l > r: return 0
                return self._query(1, 0, self.size - 1, l, r)

        def can_place(d: int) -> bool:
            # 1. Map points to 1D perimeter coordinates and sort
            def to_1d(p):
                x, y = p
                if x == 0 and y < side: return y
                if y == side and x < side: return side + x
                if x == side and y > 0: return 2 * side + (side - y)
                if y == 0 and x > 0: return 3 * side + (side - x)
                # Handle corners
                if p == [0, 0]: return 0
                if p == [0, side]: return side
                if p == [side, side]: return 2 * side
                if p == [side, 0]: return 3 * side
                return -1

            sorted_points = sorted(points, key=to_1d)
            
            n = len(points)
            doubled_points = sorted_points + sorted_points

            # 2. Transform to (u, v) = (x+y, x-y) coordinates
            transformed_points = [(p[0] + p[1], p[0] - p[1]) for p in doubled_points]
            
            # 3. Coordinate compress u and v values
            all_u = sorted(list(set(p[0] for p in transformed_points)))
            all_v = sorted(list(set(p[1] for p in transformed_points)))
            u_map = {val: i for i, val in enumerate(all_u)}
            v_map = {val: i for i, val in enumerate(all_v)}
            
            num_u = len(all_u)
            num_v = len(all_v)

            # 4. DP with Segment Trees
            st_u = SegmentTree(num_u)
            st_v = SegmentTree(num_v)

            for i in range(2 * n):
                u_i, v_i = transformed_points[i]

                max_prev_len = 0
                
                # Condition: manhattan >= d  <=>  chebyshev >= d
                # <=> |u_i - u_j| >= d  OR  |v_i - v_j| >= d
                
                # Query u-tree
                u_target_le = u_i - d
                u_idx_le = bisect.bisect_right(all_u, u_target_le) - 1
                max_prev_len = max(max_prev_len, st_u.query(0, u_idx_le))
                
                u_target_ge = u_i + d
                u_idx_ge = bisect.bisect_left(all_u, u_target_ge)
                max_prev_len = max(max_prev_len, st_u.query(u_idx_ge, num_u - 1))
                
                # Query v-tree
                v_target_le = v_i - d
                v_idx_le = bisect.bisect_right(all_v, v_target_le) - 1
                max_prev_len = max(max_prev_len, st_v.query(0, v_idx_le))

                v_target_ge = v_i + d
                v_idx_ge = bisect.bisect_left(all_v, v_target_ge)
                max_prev_len = max(max_prev_len, st_v.query(v_idx_ge, num_v - 1))
                
                current_len = 1 + max_prev_len
                
                if current_len >= k:
                    return True

                u_map_idx = u_map[u_i]
                v_map_idx = v_map[v_i]
                st_u.update(u_map_idx, current_len)
                st_v.update(v_map_idx, current_len)
                
            return False

        # Binary search for the maximum possible minimum distance
        low = 0
        high = 2 * side
        ans = 0
        while low <= high:
            mid = (low + high) // 2
            if mid == 0:
                low = mid + 1
                continue
            if can_place(mid):
                ans = mid
                low = mid + 1
            else:
                high = mid - 1
        return ans