from typing import List

class Solution:
    def maximumStrength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        prefix_sum = [0] * (n + 1)

        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + nums[i]

        dp = [[float('-inf')] * (k + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        for i in range(1, n + 1):
            for j in range(1, k + 1):
                for x in range(1, i + 1):
                    strength = 0
                    for l in range(1, x + 1):
                        strength += (-1) ** (l + 1) * (prefix_sum[i] - prefix_sum[i - l]) * (x - l + 1)
                    dp[i][j] = max(dp[i][j], dp[i - x][j - 1] + strength)

        return max(dp[n][k], 0)

# Example usage:
# sol = Solution()
# print(sol.maximumStrength([1,2,3,-1,2], 3))  # Output: 22
# print(sol.maximumStrength([12,-2,-2,-2,-2], 5))  # Output: 64
# print(sol.maximumStrength([-1,-2,-3], 1))  # Output: -1