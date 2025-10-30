class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        nums.sort()
        dp = [0] * (k + 1)
        dp[0] = 1
        prefix = [0] * (k + 1)
        prefix[0] = 1
        for num in nums:
            for i in range(k, num - 1, -1):
                dp[i] = (dp[i] + dp[i - num] * num * num) % mod
                prefix[i] = (prefix[i] + dp[i]) % mod
        return prefix[k]