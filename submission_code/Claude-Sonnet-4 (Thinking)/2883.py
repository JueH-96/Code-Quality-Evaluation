class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute powers of 5 in binary representation
        powers_of_5 = set()
        power = 1
        while True:
            binary_rep = bin(power)[2:]  # Remove '0b' prefix
            if len(binary_rep) > len(s):
                break
            powers_of_5.add(binary_rep)
            power *= 5
        
        n = len(s)
        # dp[i] = minimum partitions for s[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string requires 0 partitions
        
        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                # Check if substring is beautiful:
                # 1. No leading zeros (first character is not '0')
                # 2. Is a power of 5 in binary
                if substring[0] != '0' and substring in powers_of_5:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1