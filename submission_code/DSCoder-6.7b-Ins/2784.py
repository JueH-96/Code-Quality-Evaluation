class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        nums.sort()
        n = len(nums)
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = pow2[i - 1] * 2 % mod
        dp = [0] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            dp[i] = (dp[i - 1] * nums[i - 1] % mod + pow2[i - 1] * nums[i - 1] % mod * nums[i - 1] % mod) % mod
        return dp[-1]