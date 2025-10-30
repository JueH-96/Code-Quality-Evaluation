class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        powers_of_five = [1]
        for i in range(1, 16):
            powers_of_five.append(powers_of_five[-1] * 5)
        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                if s[j:i].startswith('0'):
                    continue
                num = int(s[j:i], 2)
                if num in powers_of_five:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1