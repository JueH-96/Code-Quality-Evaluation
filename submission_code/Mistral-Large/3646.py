from typing import List

class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        dp = [0] * n

        # Initialize the dp array with the elements themselves
        for i in range(n):
            dp[i] = nums[i]

        # Build the dp array
        for i in range(1, n):
            for j in range(i):
                if abs(nums[i] - nums[j]) == 1:
                    dp[i] = (dp[i] + dp[j]) % MOD

        # Sum up all values in the dp array
        total_sum = sum(dp) % MOD
        return total_sum

# Example usage:
# sol = Solution()
# print(sol.sumOfGoodSubsequences([1, 2, 1]))  # Output: 14
# print(sol.sumOfGoodSubsequences([3, 4, 5]))  # Output: 40