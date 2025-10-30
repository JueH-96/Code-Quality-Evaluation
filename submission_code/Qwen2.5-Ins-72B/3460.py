from typing import List
from functools import lru_cache
from bisect import bisect_left

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        requirements.sort()
        req = {end: cnt for end, cnt in requirements}
        
        @lru_cache(None)
        def dp(i, j, inv):
            if i == n:
                return 1 if inv == req[n - 1] else 0
            if i in req and inv != req[i]:
                return 0
            ans = 0
            for k in range(n):
                if k not in used:
                    used.add(k)
                    new_inv = inv + bisect_left(used, k)
                    ans += dp(i + 1, j, new_inv)
                    used.remove(k)
            return ans % MOD
        
        used = set()
        return dp(0, 0, 0)