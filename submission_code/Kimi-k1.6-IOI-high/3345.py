class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        filtered = [x for x in nums if x <= k]
        n = len(nums)
        dp = [0] * (k + 1)
        dp[0] = 1  # Base case: empty subset with sum 0
        inv2 = pow(2, MOD - 2, MOD)  # Modular inverse of 2
        
        for x in filtered:
            # Iterate backwards to avoid overwriting values we need to use
            for s in range(k, -1, -1):
                if dp[s] and s + x <= k:
                    dp[s + x] = (dp[s + x] + dp[s] * inv2) % MOD
        
        ans = dp[k] * pow(2, n, MOD) % MOD
        return ans