class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        beautiful_strings = {
            "1", "101", "11001", "1111101", "1001110001",
            "110000110101", "11110100001001"
        }
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        for i in range(n - 1, -1, -1):
            for length in [1, 3, 5, 7, 10, 12, 14]:
                if i + length <= n:
                    substring = s[i:i + length]
                    if substring in beautiful_strings:
                        dp[i] = min(dp[i], 1 + dp[i + length])
        return -1 if dp[0] == float('inf') else int(dp[0])