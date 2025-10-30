from typing import List

class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1  # Empty multiset
        
        for num in nums:
            for i in range(max_sum, num - 1, -1):
                dp[i] = (dp[i] + dp[i - num]) % MOD
        
        return sum(dp[l:r + 1]) % MOD

# Example usage:
# sol = Solution()
# print(sol.countSubMultisets([1,2,2,3], 6, 6))  # Output: 1
# print(sol.countSubMultisets([2,1,4,2,7], 1, 5))  # Output: 7
# print(sol.countSubMultisets([1,2,1,3,5,2], 3, 5))  # Output: 9