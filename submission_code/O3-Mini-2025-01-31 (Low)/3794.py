from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n, m = len(skill), len(mana)
        # dp[j] will store the time finishing processing up to current wizard for potion j.
        dp = [0] * m
        
        # For the first wizard:
        dp[0] = skill[0] * mana[0]
        for j in range(1, m):
            # Wizard 0 processes potions sequentially without waiting on any previous wizard.
            dp[j] = dp[j - 1] + skill[0] * mana[j]
        
        # For other wizards i = 1..n-1:
        for i in range(1, n):
            # new_dp will be the dp values for wizard i.
            # For the first potion, we must wait for wizard i-1 to finish potion 0, then add the processing time.
            dp[0] += skill[i] * mana[0]
            # For each subsequent potion j:
            for j in range(1, m):
                # The start time for wizard i for potion j is the later of:
                # - Wizard i finishing potion (j-1) (dp[j-1] for current wizard)
                # - Wizard (i-1) finishing potion j (dp[j] from previous row, stored in dp[j] at start of loop)
                # Then add the processing portion skill[i]*mana[j]
                dp[j] = max(dp[j - 1], dp[j]) + skill[i]*mana[j]
        
        return dp[m - 1]