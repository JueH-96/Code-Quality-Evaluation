class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        dp = {}
        total_sum = 0

        for num in nums:
            current_sum = num
            if num - 1 in dp:
                current_sum = (current_sum + (num * dp[num - 1]) % MOD) % MOD
            else:
                current_sum = current_sum % MOD

            if num in dp:
                dp[num] = (dp[num] + current_sum) % MOD
            else:
                dp[num] = current_sum

        for key in dp:
            total_sum = (total_sum + dp[key]) % MOD

        return total_sum