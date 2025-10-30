from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        # Create u and v coordinates
        u_coords = [0] * n
        v_coords = [0] * n
        for i in range(n):
            x, y = points[i]
            u_coords[i] = x + y
            v_coords[i] = x - y
        
        # Find global max and min for u and v
        global_u_max = max(u_coords)
        global_u_min = min(u_coords)
        global_v_max = max(v_coords)
        global_v_min = min(v_coords)
        global_diam = max(global_u_max - global_u_min, global_v_max - global_v_min)
        
        # Find if there are unique extremes
        # For u max
        num_u_max = sum(1 for u in u_coords if u == global_u_max)
        if num_u_max == 1:
            idx_u_max = u_coords.index(global_u_max)
        else:
            idx_u_max = -1
        
        # For u min
        num_u_min = sum(1 for u in u_coords if u == global_u_min)
        if num_u_min == 1:
            idx_u_min = u_coords.index(global_u_min)
        else:
            idx_u_min = -1
        
        # For v max
        num_v_max = sum(1 for v in v_coords if v == global_v_max)
        if num_v_max == 1:
            idx_v_max = v_coords.index(global_v_max)
        else:
            idx_v_max = -1
        
        # For v min
        num_v_min = sum(1 for v in v_coords if v == global_v_min)
        if num_v_min == 1:
            idx_v_min = v_coords.index(global_v_min)
        else:
            idx_v_min = -1
        
        # Collect candidate indices
        candidates = set()
        if idx_u_max != -1:
            candidates.add(idx_u_max)
        if idx_u_min != -1:
            candidates.add(idx_u_min)
        if idx_v_max != -1:
            candidates.add(idx_v_max)
        if idx_v_min != -1:
            candidates.add(idx_v_min)
        
        # Initialize answer to global diameter
        ans = global_diam
        
        # For each candidate, compute diameter after removal and update ans
        for idx in candidates:
            # Compute max and min u and v excluding idx
            max_u_rem = float('-inf')
            min_u_rem = float('inf')
            max_v_rem = float('-inf')
            min_v_rem = float('inf')
            for i in range(n):
                if i != idx:
                    u_val = u_coords[i]
                    v_val = v_coords[i]
                    if u_val > max_u_rem:
                        max_u_rem = u_val
                    if u_val < min_u_rem:
                        min_u_rem = u_val
                    if v_val > max_v_rem:
                        max_v_rem = v_val
                    if v_val < min_v_rem:
                        min_v_rem = v_val
            range_u_rem = max_u_rem - min_u_rem
            range_v_rem = max_v_rem - min_v_rem
            diam_rem = max(range_u_rem, range_v_rem)
            if diam_rem < ans:
                ans = diam_rem
        
        return ans