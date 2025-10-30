class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary representations of powers of 5 up to length 15.
        powers_of_5 = []
        p = 1
        while True:
            binary_repr = bin(p)[2:]  # strip the "0b" prefix
            if len(binary_repr) > 15:
                break
            powers_of_5.append(binary_repr)
            p *= 5
        
        # Convert list to a set for quick lookup
        power_set = set(powers_of_5)
        
        n = len(s)
        # dp[i] = minimal number of beautiful partitions for s[:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # base case
        
        for i in range(1, n + 1):
            for j in range(i):
                # s[j:i] is the current substring we want to check
                sub = s[j:i]
                # Check if sub is a valid power of 5 in binary
                if sub in power_set:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1