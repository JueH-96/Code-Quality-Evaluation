class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        for i in range(n):
            if s[i] == '1':
                for j in range(i, n):
                    val = int(s[i:j+1], 2)
                    if val == 1 or (val % 5 == 0 and bin(val).count('1') == 1):
                        dp[j+1] = min(dp[j+1], dp[i] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1