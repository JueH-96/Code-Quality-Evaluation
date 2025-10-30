class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        beauty = [0] * (n + 1)
        beauty[0] = 1
        for i in range(1, n + 1):
            beauty[i] = beauty[i - 1] * 5

        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            cur = 0
            for j in range(i, 0, -1):
                cur = cur * 2 + s[j - 1] - ord('0')
                if cur >= beauty[j]:
                    dp[i] = min(dp[i], dp[j - 1] + 1)
                else:
                    break

        return dp[n] if dp[n] != float('inf') else -1