class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        powers = set()
        power = 1
        while power <= 1 << 15:
            powers.add(bin(power)[2:])
            power *= 5
        for i in range(1, n + 1):
            for j in range(i):
                sub = s[j:i]
                if sub in powers:
                    dp[i] = min(dp[i], dp[j] + 1)
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]