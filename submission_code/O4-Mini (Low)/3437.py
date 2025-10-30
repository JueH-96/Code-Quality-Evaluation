from typing import List

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Count total damage per distinct power value
        from collections import Counter
        cnt = Counter(power)
        
        # Sort unique power levels
        vals = sorted(cnt.keys())
        # Precompute the total damage accumulated if we pick all spells of a given power
        total_damage = [v * cnt[v] for v in vals]
        
        n = len(vals)
        # dp[i] = max damage we can achieve considering the first i+1 distinct power levels
        dp = [0] * n
        
        left = 0  # sliding pointer to find the rightmost non-conflicting index
        
        for i in range(n):
            # Move left so that vals[left] is the first value >= vals[i] - 2
            while left < i and vals[left] < vals[i] - 2:
                left += 1
            # j is the last index < i that does NOT conflict with i
            j = left - 1
            
            # Option 1: skip current power level -> dp[i-1]
            skip_i = dp[i-1] if i > 0 else 0
            # Option 2: take all spells of current power level -> total_damage[i] + dp[j]
            take_i = total_damage[i] + (dp[j] if j >= 0 else 0)
            
            dp[i] = max(skip_i, take_i)
        
        return dp[-1] if dp else 0