class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        powers_of_5 = []
        for i in range(16):
            power_of_5 = 5**i
            binary_representation = bin(power_of_5)[2:]
            if len(binary_representation) <= 15:
                powers_of_5.append(binary_representation)
            else:
                break
        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0

        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if substring.startswith('0'):
                    continue
                if substring in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]