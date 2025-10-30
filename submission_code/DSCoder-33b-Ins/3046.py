class Solution:
    def minimumOperations(self, num: str) -> int:
        n = len(num)
        dp = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            dp[i] = dp[i + 1]
            if num[i] == '0':
                dp[i] += 1
            else:
                for j in range(i + 1, n + 1):
                    if int(num[i:j]) % 25 == 0:
                        dp[i] = min(dp[i], 1 + dp[j])
        return dp[0]