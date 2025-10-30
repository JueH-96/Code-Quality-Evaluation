from typing import List
from collections import defaultdict

class Solution:
    def maximumTotalDamage(self, power: List[int]) -> int:
        # Sort the power array
        power.sort()

        # Dictionary to store the sum of damages for each unique damage value
        damage_sum = defaultdict(int)
        for p in power:
            damage_sum[p] += p

        # Convert the dictionary to a sorted list of tuples
        damage_list = sorted(damage_sum.items())

        # Initialize the DP array
        n = len(damage_list)
        if n == 0:
            return 0

        dp = [0] * n
        dp[0] = damage_list[0][1]

        # Fill the DP array
        for i in range(1, n):
            damage, total = damage_list[i]
            # Include the current damage
            include_current = total + (dp[i-1] if i-1 >= 0 else 0)
            # Exclude the current damage and check the previous valid damage
            exclude_current = dp[i-1]
            for j in range(i-1, -1, -1):
                if damage_list[j][0] < damage - 2:
                    exclude_current = max(exclude_current, dp[j])
                    break
            dp[i] = max(include_current, exclude_current)

        return dp[-1]

# Example usage:
# solution = Solution()
# print(solution.maximumTotalDamage([1,1,3,4]))  # Output: 6
# print(solution.maximumTotalDamage([7,1,6,6]))  # Output: 13