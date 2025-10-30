class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = {}
        total_sum = 0
        for num in nums:
            curr_sum = num
            curr_count = 1
            if num - 1 in dp:
                curr_sum = (curr_sum + dp[num - 1][0] * num + dp[num - 1][1]) % MOD
                curr_count = (curr_count + dp[num - 1][1]) % MOD
            if num + 1 in dp:
                curr_sum = (curr_sum + dp[num + 1][0] * num + dp[num + 1][1]) % MOD
                curr_count = (curr_count + dp[num + 1][1]) % MOD
            
            dp[num] = (curr_sum, curr_count)
            total_sum = (total_sum + curr_sum) % MOD
        return total_sum