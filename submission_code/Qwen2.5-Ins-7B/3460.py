from typing import List
from functools import lru_cache

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        
        @lru_cache(None)
        def dp(end, remaining):
            if end == n:
                return 1
            if not remaining:
                return 0
            total = 0
            for i in range(len(remaining)):
                if remaining[i][1] > 0:
                    remaining[i][1] -= 1
                    total += dp(end + 1, tuple(sorted(remaining, key=lambda x: x[1])))
                    remaining[i][1] += 1
            return total % MOD
        
        return dp(0, tuple(sorted(requirements, key=lambda x: x[0])))