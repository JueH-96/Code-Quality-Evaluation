from typing import List

class Solution:
    def countPairs(self, coordinates: List[List[int]], k: int) -> int:
        # If k == 0, we only want pairs of identical points.
        if k == 0:
            from collections import Counter
            freq = Counter((x, y) for x, y in coordinates)
            res = 0
            for v in freq.values():
                res += v * (v - 1) // 2
            return res

        # Otherwise, pack each (x,y) into a single integer key to speed up lookups.
        # We know 0 <= x,y <= 10^6 < 2^20, so shifting by 20 bits is safe.
        SHIFT = 20
        cnt = {}
        for x, y in coordinates:
            key = (x << SHIFT) | y
            cnt[key] = cnt.get(key, 0) + 1

        res = 0
        get_cnt = cnt.get
        # For each point, try all dx from 0..k, set dy = k - dx,
        # and look up how many points match (x^dx, y^dy).
        for x, y in coordinates:
            for dx in range(k + 1):
                dy = k - dx
                tx = x ^ dx
                ty = y ^ dy
                tkey = (tx << SHIFT) | ty
                res += get_cnt(tkey, 0)

        # Each pair (i,j) was counted twice (once from i's side and once from j's side).
        return res // 2