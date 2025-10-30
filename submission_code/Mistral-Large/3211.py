from typing import List

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        if not nums:
            return 0

        # Dynamic programming array to store the length of the longest increasing subsequence ending at each index
        dp = [1] * len(nums)

        # Compute the LIS ending at each index
        for i in range(1, len(nums)):
            for j in range(i):
                if nums[i] >= nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)

        # The maximum value in dp array is the length of the LIS
        return max(dp)

# Example usage:
# solution = Solution()
# print(solution.findMaximumLength([5, 2, 2]))  # Output: 1
# print(solution.findMaximumLength([1, 2, 3, 4]))  # Output: 4
# print(solution.findMaximumLength([4, 3, 2, 6]))  # Output: 3