from typing import List

class Solution:
    def maxTotalReward(self, rewardValues: List[int]) -> int:
        # Sort the values – we will try to decide, in this order, 
        # whether we take each reward or not.
        rewardValues.sort()
        
        # All individual rewards are ≤ 2000.
        # Before we pick a new reward our running sum must be < that reward,
        # hence strictly < 2000.  After the very last pick the sum can be
        # at most 1999 + 2000 = 3999, so keeping states up to 4000 is enough.
        MAX_SUM = 4000
        dp = [False] * (MAX_SUM + 1)   # dp[s] == True  →  total 's' is achievable
        dp[0] = True                   # we can always start from 0
        
        current_max = 0                # largest sum seen so far
        
        for v in rewardValues:
            # To take v we need current sum s < v, therefore only
            # sums below v (and in any case below 2000) matter here.
            upper = min(v - 1, 1999)
            
            # iterate backwards so every reward is used at most once
            for s in range(upper, -1, -1):
                if dp[s]:
                    new_sum = s + v
                    dp[new_sum] = True
                    if new_sum > current_max:
                        current_max = new_sum
        
        # answer: the greatest reachable total
        for s in range(MAX_SUM, -1, -1):
            if dp[s]:
                return s
        return 0