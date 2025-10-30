from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        power.sort()
        dp = {}
        
        def dfs(i, prev):
            if i == len(power):
                return 0
            if (i, prev) in dp:
                return dp[(i, prev)]
            
            skip = dfs(i + 1, prev)
            take = 0
            if abs(power[i] - prev) > 1:
                take = power[i] + dfs(i + 1, power[i])
            
            dp[(i, prev)] = max(skip, take)
            return dp[(i, prev)]
        
        return dfs(0, -float('inf'))