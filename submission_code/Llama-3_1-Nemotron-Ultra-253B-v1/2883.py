class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        beautiful = {"1", "101", "11001", "1111101", "1001110001", "110000110101", "11110100001001"}
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(1, n + 1):
            for b in beautiful:
                len_b = len(b)
                if i >= len_b and s[i - len_b:i] == b:
                    dp[i] = min(dp[i], dp[i - len_b] + 1)
        return dp[n] if dp[n] != float('inf') else -1