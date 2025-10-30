class Solution:
    def sumOfPower(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        nums.sort()
        dp = [0] * (n + 1)
        pre = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = (nums[i] + pre[i + 1]) % mod
            pre[i] = (pre[i + 1] + dp[i]) % mod
        return sum(dp) % mod