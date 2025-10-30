class Solution:
    def numberOfWays(self, n: int, x: int) -> int:
        mod = 10 ** 9 + 7
        max_i = int(n ** (1 / x)) + 1
        nums = []
        for i in range(1, max_i+1):
            num = i ** x
            if num > n:
                break
            nums.append(num)
        dp = [0] * (n + 1)
        dp[0] = 1
        for num in nums:
            for s in range(n, num - 1, -1):
                dp[s] = (dp[s] + dp[s - num]) % mod
        return dp[n]