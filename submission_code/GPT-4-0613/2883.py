class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_five = [int(bin(5**i)[2:]) for i in range(13)]
        dp = [float('inf')] * (len(s) + 1)
        dp[0] = 0
        for i in range(1, len(s) + 1):
            if s[i-1] == '0':
                continue
            for power in powers_of_five:
                if len(str(power)) <= i and int(s[i-len(str(power)):i]) == power:
                    dp[i] = min(dp[i], dp[i-len(str(power))] + 1)
        return dp[-1] if dp[-1] != float('inf') else -1