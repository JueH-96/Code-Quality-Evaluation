from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        # We use four "projections":
        # f0 = x + y, f1 = x - y, f2 = -x + y, f3 = -x - y.
        # The max Manhattan distance is max_k (max(fk) - min(fk)).
        # When removing one point, only if it was an extremal in some fk
        # will the extremum change (to the 2nd best). So we track for each
        # fk the top two max and min values and their indices.
        
        n = len(points)
        # For each k in [0..3], we store:
        # max1_val[k], max1_idx[k], max2_val[k], max2_idx[k]
        # min1_val[k], min1_idx[k], min2_val[k], min2_idx[k]
        max1_val = [-10**30]*4
        max1_idx = [-1]*4
        max2_val = [-10**30]*4
        max2_idx = [-1]*4
        min1_val = [10**30]*4
        min1_idx = [-1]*4
        min2_val = [10**30]*4
        min2_idx = [-1]*4
        
        # helper to compute fk
        def fk(k, x, y):
            if k == 0: return x + y
            if k == 1: return x - y
            if k == 2: return -x + y
            return -x - y
        
        # First pass: identify top two max/min for each fk
        for i, (x, y) in enumerate(points):
            for k in range(4):
                v = fk(k, x, y)
                # update maxes
                if v > max1_val[k]:
                    # demote old max1 to max2
                    max2_val[k], max2_idx[k] = max1_val[k], max1_idx[k]
                    max1_val[k], max1_idx[k] = v, i
                elif v > max2_val[k]:
                    max2_val[k], max2_idx[k] = v, i
                # update mins
                if v < min1_val[k]:
                    # demote old min1 to min2
                    min2_val[k], min2_idx[k] = min1_val[k], min1_idx[k]
                    min1_val[k], min1_idx[k] = v, i
                elif v < min2_val[k]:
                    min2_val[k], min2_idx[k] = v, i
        
        # Collect candidate points whose removal might affect an extremum
        candidates = set()
        for k in range(4):
            for idx in (max1_idx[k], max2_idx[k], min1_idx[k], min2_idx[k]):
                if idx != -1:
                    candidates.add(idx)
        
        # For each candidate, compute the resulting max distance
        ans = 10**30
        for i in candidates:
            # For each k, figure out the resulting max/min after removing i
            cur_max_dist = 0
            for k in range(4):
                if i == max1_idx[k]:
                    cur_max = max2_val[k]
                else:
                    cur_max = max1_val[k]
                if i == min1_idx[k]:
                    cur_min = min2_val[k]
                else:
                    cur_min = min1_val[k]
                cur_max_dist = max(cur_max_dist, cur_max - cur_min)
            ans = min(ans, cur_max_dist)
        
        return ans