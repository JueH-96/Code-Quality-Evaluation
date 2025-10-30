class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary strings of powers of 5 up to 5^6 (since 5^7 exceeds 2^15)
        valid_powers = set()
        current = 1  # 5^0
        valid_powers.add(bin(current)[2:])  # binary '1'
        for _ in range(6):
            current *= 5
            valid_powers.add(bin(current)[2:])
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case: 0 characters require 0 substrings
        
        for i in range(1, n + 1):
            for j in range(i):
                substr = s[j:i]
                # Skip substrings with leading zeros
                if substr[0] == '0':
                    continue
                if substr in valid_powers:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1