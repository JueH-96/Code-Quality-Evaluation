from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        # Initialize the dp array with -1 for all sums except 0 which is 0
        dp = [-1] * (target + 1)
        dp[0] = 0
        
        # Update dp array for each number in nums
        for num in nums:
            # Traverse in reverse to avoid using the same number twice
            for sub_target in range(target, num - 1, -1):
                # If the sub_target - num has a subsequence, update the dp array
                if dp[sub_target - num] != -1:
                    dp[sub_target] = max(dp[sub_target], dp[sub_target - num] + 1)
        
        return dp[target]

# Example usage:
# sol = Solution()
# print(sol.lengthOfLongestSubsequence([1,2,3,4,5], 9))  # Output: 3
# print(sol.lengthOfLongestSubsequence([4,1,3,2,1,5], 7))  # Output: 4
# print(sol.lengthOfLongestSubsequence([1,1,5,4,5], 3))  # Output: -1