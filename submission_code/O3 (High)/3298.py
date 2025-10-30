from typing import List
from collections import defaultdict

class Solution:
    def maxSelectedElements(self, nums: List[int]) -> int:
        # sort the numbers so that we always process smaller (or equal) originals first
        nums.sort()
        
        # dp[value] = maximum sequence length whose largest (last) chosen value is `value`
        dp = dict()          # we just need a normal dictionary
        best = 0             # answer to return
        
        for x in nums:
            # values from dp BEFORE we touch it in this iteration
            prev_for_x_minus_1 = dp.get(x - 1, 0)   # sequence ending with x-1
            prev_for_x         = dp.get(x,     0)   # sequence ending with x
            
            # use the current element as `x`  (no increment)
            cand_using_x   = prev_for_x_minus_1 + 1
            if cand_using_x > dp.get(x, 0):
                dp[x] = cand_using_x
                best = max(best, cand_using_x)
            
            # use the current element as `x + 1` (increment by one)
            cand_using_x1  = prev_for_x + 1
            nxt = x + 1
            if cand_using_x1 > dp.get(nxt, 0):
                dp[nxt] = cand_using_x1
                best = max(best, cand_using_x1)
        
        return best