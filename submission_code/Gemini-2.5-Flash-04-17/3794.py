from typing import List

class Solution:
    def minTime(self, skill: List[int], mana: List[int]) -> int:
        n = len(skill)
        m = len(mana)

        # Precompute prefix sums of skill
        # prefix_skill_sum[i] = sum(skill[0]...skill[i-1])
        # prefix_skill_sum[0] = 0
        # prefix_skill_sum[1] = skill[0]
        # prefix_skill_sum[n] = sum(skill[0]...skill[n-1])
        prefix_skill_sum = [0] * (n + 1)
        for i in range(n):
            prefix_skill_sum[i+1] = prefix_skill_sum[i] + skill[i]

        # prev_dp[i] stores the finish time of wizard i on the previous potion (j-1)
        prev_dp = [0] * n

        # Calculate finish times for the first potion (j=0)
        # Start time of potion 0 at wizard 0 is 0.
        # Finish time for wizard i on potion 0 = StartP_0 + sum(skill[0]...skill[i]) * mana[0]
        # StartP_0 = 0
        # sum(skill[0]...skill[i]) is prefix_skill_sum[i+1] using our definition
        for i in range(n):
             prev_dp[i] = prefix_skill_sum[i+1] * mana[0]

        # Iterate through potions from the second one (j=1 to m-1)
        for j in range(1, m):
            # Calculate the minimum start time for potion j at wizard 0 (StartP_j)
            # The strict "immediate pass" rule implies start[i][j] = finish[i-1][j] for i > 0.
            # If this holds, finish[i][j] = StartP_j + sum(skill[0]...skill[i]) * mana[j]
            # = StartP_j + prefix_skill_sum[i+1] * mana[j].
            # For this strict chain to be possible, wizard i must be free when potion j arrives at i.
            # Potion j arrives at i at time finish[i-1][j]. Wizard i is free at prev_dp[i].
            # So, we need finish[i-1][j] >= prev_dp[i] for i = 1...n-1.
            # Substituting the formula for finish[i-1][j]:
            # StartP_j + sum(skill[0]...skill[i-1]) * mana[j] >= prev_dp[i]
            # StartP_j + prefix_skill_sum[i] * mana[j] >= prev_dp[i]
            # StartP_j >= prev_dp[i] - prefix_skill_sum[i] * mana[j] for i = 1...n-1.
            # Also, wizard 0 must finish potion j-1 before starting potion j: StartP_j >= prev_dp[0].
            # This can be written as StartP_j >= prev_dp[0] - prefix_skill_sum[0] * mana[j] since prefix_skill_sum[0] is 0.
            # Combining, StartP_j >= prev_dp[i] - prefix_skill_sum[i] * mana[j] for i = 0...n-1.
            # The minimum StartP_j is the maximum of these lower bounds.

            StartP_j = 0 # Minimum possible start time is 0
            for i in range(n):
                 StartP_j = max(StartP_j, prev_dp[i] - prefix_skill_sum[i] * mana[j])

            # Calculate finish times for the current potion (j) assuming the strict chain holds
            curr_dp = [0] * n
            # Finish time of wizard i on potion j = StartP_j + sum(skill[0]...skill[i]) * mana[j]
            # sum(skill[0]...skill[i]) is prefix_skill_sum[i+1]
            for i in range(n):
                curr_dp[i] = StartP_j + prefix_skill_sum[i+1] * mana[j]

            # Update prev_dp for the next iteration
            prev_dp = curr_dp

        # The final answer is the finish time of the last wizard on the last potion
        return prev_dp[n-1]