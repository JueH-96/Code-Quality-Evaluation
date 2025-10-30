class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        max_sum = sum(nums)
        dp = [0] * (max_sum + 1)
        dp[0] = 1

        for num in nums:
            for j in range(max_sum, num - 1, -1):
                dp[j] = (dp[j] + dp[j - num]) % MOD

        return sum(dp[l:r+1]) % MOD