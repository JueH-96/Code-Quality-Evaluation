class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        dp = [0] * (k + 1)
        dp[0] = 1
        inv2 = pow(2, mod - 2, mod)
        
        for num in nums:
            new_dp = dp.copy()
            if num <= k:
                for s in range(k, num - 1, -1):
                    new_dp[s] = (new_dp[s] + dp[s - num] * inv2) % mod
            dp = new_dp
        
        total_factor = pow(2, n, mod)
        return (dp[k] * total_factor) % mod