class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers = []
        for i in range(19):
            powers.append(bin(5**i)[2:])

        n = len(s)
        dp = [-1] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for p in powers:
                lp = len(p)
                if lp <= i and s[i-lp:i] == p and (lp == 1 or s[i-lp] == '1') and dp[i-lp] != -1:
                    if dp[i] == -1:
                        dp[i] = dp[i - lp] + 1
                    else:
                        dp[i] = min(dp[i], dp[i - lp] + 1)
        
        return dp[n]