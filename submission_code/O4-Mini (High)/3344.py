class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # We transform each point (x, y) into u = x + y and v = x - y.
        # The Manhattan diameter of a set is max(max_u - min_u, max_v - min_v).
        # Removing a point only affects extremes if it is one of the extrema.
        INF = 10**18
        # Track top two max/min for u = x + y
        u_max1_val = u_max2_val = -INF
        u_max1_idx = u_max2_idx = -1
        u_min1_val = u_min2_val = INF
        u_min1_idx = u_min2_idx = -1
        # Track top two max/min for v = x - y
        v_max1_val = v_max2_val = -INF
        v_max1_idx = v_max2_idx = -1
        v_min1_val = v_min2_val = INF
        v_min1_idx = v_min2_idx = -1

        # First pass: compute the two largest and two smallest values of u and v
        for i, (x, y) in enumerate(points):
            ui = x + y
            # update u max
            if ui > u_max1_val:
                u_max2_val, u_max2_idx = u_max1_val, u_max1_idx
                u_max1_val, u_max1_idx = ui, i
            elif ui > u_max2_val:
                u_max2_val, u_max2_idx = ui, i
            # update u min
            if ui < u_min1_val:
                u_min2_val, u_min2_idx = u_min1_val, u_min1_idx
                u_min1_val, u_min1_idx = ui, i
            elif ui < u_min2_val:
                u_min2_val, u_min2_idx = ui, i

            vi = x - y
            # update v max
            if vi > v_max1_val:
                v_max2_val, v_max2_idx = v_max1_val, v_max1_idx
                v_max1_val, v_max1_idx = vi, i
            elif vi > v_max2_val:
                v_max2_val, v_max2_idx = vi, i
            # update v min
            if vi < v_min1_val:
                v_min2_val, v_min2_idx = v_min1_val, v_min1_idx
                v_min1_val, v_min1_idx = vi, i
            elif vi < v_min2_val:
                v_min2_val, v_min2_idx = vi, i

        # Only points that are at one of the four extrema can reduce the diameter when removed.
        candidates = {
            u_max1_idx, u_max2_idx, u_min1_idx, u_min2_idx,
            v_max1_idx, v_max2_idx, v_min1_idx, v_min2_idx
        }

        ans = INF
        # Try removing each candidate and recompute the diameter from stored extremes
        for k in candidates:
            if k < 0:
                continue
            # New u-extrema after removing k
            max_u = u_max2_val if k == u_max1_idx else u_max1_val
            min_u = u_min2_val if k == u_min1_idx else u_min1_val
            # New v-extrema after removing k
            max_v = v_max2_val if k == v_max1_idx else v_max1_val
            min_v = v_min2_val if k == v_min1_idx else v_min1_val
            # The resulting diameter
            cur = max(max_u - min_u, max_v - min_v)
            if cur < ans:
                ans = cur

        return ans