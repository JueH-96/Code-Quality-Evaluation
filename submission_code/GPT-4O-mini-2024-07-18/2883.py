class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all powers of 5 in binary representation up to 15 bits
        beautiful_numbers = set()
        power = 1
        while True:
            binary_representation = bin(power)[2:]  # Get binary representation without '0b' prefix
            if len(binary_representation) > 15:
                break
            beautiful_numbers.add(binary_representation)
            power *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 substrings for an empty string
        
        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if substring in beautiful_numbers and (substring[0] != '0' or len(substring) == 1):
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1