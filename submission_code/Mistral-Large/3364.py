from typing import List

class Solution:
    def minimumValueSum(self, nums: List[int], andValues: List[int]) -> int:
        n = len(nums)
        m = len(andValues)

        # Prefix AND array
        prefix_and = [0] * (n + 1)
        for i in range(n):
            prefix_and[i + 1] = prefix_and[i] & nums[i]

        # DP table initialization
        dp = [[float('inf')] * (m + 1) for _ in range(n + 1)]
        dp[0][0] = 0

        # Fill the DP table
        for i in range(1, n + 1):
            for j in range(1, min(i, m) + 1):
                for k in range(i):
                    if prefix_and[i] & nums[k] == andValues[j - 1]:
                        dp[i][j] = min(dp[i][j], dp[k][j - 1] + nums[i - 1])

        # Check if it's possible to divide nums into m subarrays
        if dp[n][m] == float('inf'):
            return -1
        else:
            return dp[n][m]

# Example usage:
# sol = Solution()
# print(sol.minimumValueSum([1,4,3,3,2], [0,3,3,2]))  # Output: 12
# print(sol.minimumValueSum([2,3,5,7,7,7,5], [0,7,5]))  # Output: 17
# print(sol.minimumValueSum([1,2,3,4], [2]))  # Output: -1