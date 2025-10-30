class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute powers of 5 and their binary representations
        powers_of_5_binary = set()
        power = 1
        # Since s.length <= 15, we need powers of 5 up to 2^15
        while power < (1 << 15):
            powers_of_5_binary.add(bin(power)[2:])  # Remove '0b' prefix
            power *= 5
        
        def is_beautiful(substring):
            # Check if substring has leading zeros
            if substring[0] == '0':
                return False
            # Check if it's a power of 5
            return substring in powers_of_5_binary
        
        n = len(s)
        # dp[i] represents minimum partitions needed for s[i:]
        # Use n+1 to represent impossible case
        dp = [float('inf')] * (n + 1)
        dp[n] = 0  # Empty string needs 0 partitions
        
        # Work backwards from the end
        for i in range(n - 1, -1, -1):
            # Try all possible substrings starting at position i
            for j in range(i + 1, n + 1):
                substring = s[i:j]
                if is_beautiful(substring):
                    dp[i] = min(dp[i], 1 + dp[j])
        
        return dp[0] if dp[0] != float('inf') else -1