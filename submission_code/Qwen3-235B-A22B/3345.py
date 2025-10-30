class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        inv2 = pow(2, MOD - 2, MOD)
        n = len(nums)
        dp = [0] * (k + 1)
        dp[0] = 1  # Initial state: empty subset contributes 1 to sum 0
        
        for num in nums:
            if num > k:
                continue
            # Traverse from k down to num to prevent overcounting
            for s in range(k, num - 1, -1):
                dp[s] = (dp[s] + dp[s - num] * inv2) % MOD
        
        ans = (pow(2, n, MOD) * dp[k]) % MOD
        return ans