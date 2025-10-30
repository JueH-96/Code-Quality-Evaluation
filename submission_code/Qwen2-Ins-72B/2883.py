class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5 = [bin(5 ** i)[2:] for i in range(7)]
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0
        for i in range(1, len(s) + 1):
            for p in powers_of_5:
                if s[:i].endswith(p):
                    dp[i] = min(dp[i], dp[i - len(p)] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1