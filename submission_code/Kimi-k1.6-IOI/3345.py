class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        inv2 = pow(2, MOD - 2, MOD)
        dp = [0] * (k + 1)
        dp[0] = 1  # Base case: empty subsequence
        
        for num in nums:
            for s in range(k, -1, -1):
                if dp[s] and s + num <= k:
                    dp[s + num] = (dp[s + num] + dp[s] * inv2) % MOD
        
        return (dp[k] * pow(2, len(nums), MOD)) % MOD