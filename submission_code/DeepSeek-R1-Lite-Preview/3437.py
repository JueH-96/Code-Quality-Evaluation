from typing import List
import bisect

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        from collections import Counter

        # Step 1: Count frequency and calculate total_damage for each unique power
        count = Counter(power)
        groups = sorted([(value, count[value]) for value in count], key=lambda x: x[0])
        
        # Extract sorted unique power values and their total damages
        sorted_values = [group[0] for group in groups]
        total_damage = [group[0] * group[1] for group in groups]
        
        n = len(groups)
        if n == 0:
            return 0
        
        # Initialize dp array
        dp = [0] * n
        dp[0] = total_damage[0]
        
        # Step 3: Fill dp array
        for i in range(1, n):
            value = sorted_values[i]
            # Find the last group where value <= current value - 3
            target = value - 3
            # Find the rightmost group with value <= target
            j = bisect.bisect_right(sorted_values, target) - 1
            if j >= 0:
                dp[i] = max(dp[i-1], dp[j] + total_damage[i])
            else:
                dp[i] = max(dp[i-1], total_damage[i])
        
        # The answer is the maximum in dp
        return dp[-1]