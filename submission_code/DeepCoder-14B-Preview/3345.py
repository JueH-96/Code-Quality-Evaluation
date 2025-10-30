MOD = 10**9 + 7

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_sum = k  # We only care about sums up to k
        
        # Initialize DP table
        dp = [[0] * (max_sum + 1) for _ in range(n + 1)]
        dp[0][0] = 1
        
        for num in nums:
            # Iterate m and s in reverse to avoid reusing the same element multiple times in the same subset
            for m in reversed(range(n + 1)):
                for s in reversed(range(max_sum + 1)):
                    if dp[m][s]:
                        new_m = m + 1
                        new_s = s + num
                        if new_m <= n and new_s <= max_sum:
                            dp[new_m][new_s] = (dp[new_m][new_s] + dp[m][s]) % MOD
        
        total = 0
        for m in range(1, n + 1):
            count = dp[m][k]
            exponent = n - m
            pow_val = pow(2, exponent, MOD)
            total = (total + count * pow_val) % MOD
        
        return total % MOD