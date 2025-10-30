from typing import List
from functools import lru_cache
import math

class Solution:
    def findMinimumTime(self, strength: List[int], K: int) -> int:
        """
        Dynamic-programming over subsets (bitmask).
        n ≤ 8  ⇒  at most 2⁸ = 256 states – extremely small.
        
        DP(mask)  =  minimal additional minutes needed to break every lock
                     that is **not** yet broken in the bitmask *mask*
                     (bit i = 1  ⇨  lock i already broken).
                     
        Let cnt = number of bits already set in mask.
        Sword factor just before breaking the next lock is 
        
                X = 1 + cnt * K
        
        For every still-closed lock j we can break it after
        
                ceil(strength[j] / X)  minutes,
                
        then recurse on the new mask with bit j set.
        """
        
        n = len(strength)
        full_mask = (1 << n) - 1      # all locks opened
        
        @lru_cache(maxsize=None)
        def dp(mask: int) -> int:
            if mask == full_mask:     # all locks are already broken
                return 0
            
            cnt_opened = bin(mask).count('1')
            X = 1 + cnt_opened * K
            best = math.inf
            
            for i in range(n):
                if not (mask & (1 << i)):              # lock i still closed
                    minutes = (strength[i] + X - 1) // X   # ceil division
                    best = min(best, minutes + dp(mask | (1 << i)))
            
            return best
        
        return dp(0)