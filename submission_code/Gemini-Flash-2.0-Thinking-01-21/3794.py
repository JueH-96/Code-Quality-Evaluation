from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)

        # dp[i] will store the finish time of wizard i for the previous potion
        # Initially, all wizards are free at time 0 (before any potions)
        dp = [0] * n

        # Iterate through each potion
        for j in range(m):
            # new_dp will store the finish times for the current potion j
            new_dp = [0] * n

            # Calculate finish time for the first wizard (wizard 0) on potion j
            # Wizard 0 starts working on potion j as soon as they finish potion j-1 (at dp[0])
            # Potion j becomes available at Wizard 0 at some time. The earliest it can be available is 0 for Potion 0.
            # For subsequent potions, the problem statement implies they are processed in order.
            # The most straightforward interpretation is that Potion j is available at Wizard 0 as soon as Wizard 0 finishes Potion j-1.
            duration_0j = skill[0] * mana[j]
            new_dp[0] = dp[0] + duration_0j

            # Calculate finish times for remaining wizards on potion j
            for i in range(1, n):
                duration_ij = skill[i] * mana[j]
                
                # Wizard i can start working on potion j only when:
                # 1. Wizard i-1 finishes potion j (at new_dp[i-1]) -> potion j arrives at wizard i
                # 2. Wizard i finishes potion j-1 (at dp[i]) -> wizard i becomes free
                start_time_ij = max(new_dp[i-1], dp[i])
                
                # Finish time is start time plus duration
                new_dp[i] = start_time_ij + duration_ij

            # Update dp to be the finish times for the potion j
            dp = new_dp

        # The minimum time required is the time the last wizard finishes the last potion
        return dp[n-1]