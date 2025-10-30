class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        powers = set()
        power = 1
        while power <= (1 << n) - 1:
            powers.add(bin(power)[2:])
            power *= 5

        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(max(0, i - 15), i):
                substring = s[j:i]
                if substring in powers:
                    dp[i] = min(dp[i], dp[j] + 1)

        return dp[n] if dp[n] != float('inf') else -1