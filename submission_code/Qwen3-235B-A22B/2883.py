class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all valid binary strings of powers of 5 up to 5^k where binary length <= 15
        valid = set()
        current_power = 1
        while True:
            bin_str = bin(current_power)[2:]  # Convert current_power to binary string without '0b'
            valid.add(bin_str)
            next_power = current_power * 5
            next_str = bin(next_power)[2:]
            if len(next_str) > 15:
                break
            current_power = next_power
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters require 0 substrings
        
        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if substring in valid:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1