class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary representations of 5^k up to 15 bits
        power_of_5_bin = set()
        k = 0
        while True:
            num = 5 ** k
            bin_str = bin(num)[2:]  # Convert to binary without '0b' prefix
            if len(bin_str) > 15:
                break
            power_of_5_bin.add(bin_str)
            k += 1
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 substrings
        
        for i in range(1, n + 1):
            for j in range(i):
                # Skip substrings starting with '0'
                if s[j] == '0':
                    continue
                substr = s[j:i]
                if substr in power_of_5_bin:
                    if dp[j] + 1 < dp[i]:
                        dp[i] = dp[j] + 1
        
        return dp[n] if dp[n] != float('inf') else -1