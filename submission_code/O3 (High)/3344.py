from typing import List

class Solution:
    def minimumDistance(self, points: List[List[int]]) -> int:
        n = len(points)
        
        # Arrays with the transformed coordinates
        s_vals = [0] * n          #  s = x + y
        d_vals = [0] * n          #  d = x - y
        
        # Helpers for extremes (first / second extreme + their counts)
        NEG_INF = -10**20
        POS_INF =  10**20
        
        max_s1 = NEG_INF; max_s2 = NEG_INF; cnt_max_s1 = 0
        min_s1 = POS_INF; min_s2 = POS_INF; cnt_min_s1 = 0
        
        max_d1 = NEG_INF; max_d2 = NEG_INF; cnt_max_d1 = 0
        min_d1 = POS_INF; min_d2 = POS_INF; cnt_min_d1 = 0
        
        # Pass 1 ─ collect s, d and their extreme statistics
        for i, (x, y) in enumerate(points):
            s = x + y
            d = x - y
            s_vals[i] = s
            d_vals[i] = d
            
            # ----- for s -----
            if s > max_s1:
                max_s2, max_s1, cnt_max_s1 = max_s1, s, 1
            elif s == max_s1:
                cnt_max_s1 += 1
            elif s > max_s2:
                max_s2 = s
            
            if s < min_s1:
                min_s2, min_s1, cnt_min_s1 = min_s1, s, 1
            elif s == min_s1:
                cnt_min_s1 += 1
            elif s < min_s2:
                min_s2 = s
            
            # ----- for d -----
            if d > max_d1:
                max_d2, max_d1, cnt_max_d1 = max_d1, d, 1
            elif d == max_d1:
                cnt_max_d1 += 1
            elif d > max_d2:
                max_d2 = d
            
            if d < min_d1:
                min_d2, min_d1, cnt_min_d1 = min_d1, d, 1
            elif d == min_d1:
                cnt_min_d1 += 1
            elif d < min_d2:
                min_d2 = d
        
        # Guard: if second extremes never updated (all equal values)
        if max_s2 == NEG_INF: max_s2 = max_s1
        if min_s2 == POS_INF: min_s2 = min_s1
        if max_d2 == NEG_INF: max_d2 = max_d1
        if min_d2 == POS_INF: min_d2 = min_d1
        
        # Pass 2 ─ try removing every point
        best = POS_INF
        for s, d in zip(s_vals, d_vals):
            # new extremes of s after removing the point
            new_max_s = max_s1 if not (s == max_s1 and cnt_max_s1 == 1) else max_s2
            new_min_s = min_s1 if not (s == min_s1 and cnt_min_s1 == 1) else min_s2
            s_range = new_max_s - new_min_s
            
            # new extremes of d after removing the point
            new_max_d = max_d1 if not (d == max_d1 and cnt_max_d1 == 1) else max_d2
            new_min_d = min_d1 if not (d == min_d1 and cnt_min_d1 == 1) else min_d2
            d_range = new_max_d - new_min_d
            
            # diameter = max(range in s, range in d)
            best = min(best, max(s_range, d_range))
        
        return best