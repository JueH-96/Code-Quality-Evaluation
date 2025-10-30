class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        powers_of_5 = []
        for i in range(16):
            powers_of_5.append(bin(5**i)[2:])
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                sub = s[j:i]
                if sub[0] == '0':
                    continue
                if sub in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]