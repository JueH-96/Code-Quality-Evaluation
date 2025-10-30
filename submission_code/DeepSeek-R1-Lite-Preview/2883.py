class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute powers of 5 up to the maximum possible value for a 15-bit binary string
        powers_of_five = []
        power = 1
        while power <= (1 << 15) - 1:
            powers_of_five.append(power)
            power *= 5
        
        n = len(s)
        # Initialize dp array with infinity
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 substrings
        
        # Fill the dp array
        for i in range(1, n + 1):
            for j in range(i):
                substring = s[j:i]
                if substring[0] == '1':
                    decimal_value = int(substring, 2)
                    if decimal_value in powers_of_five:
                        dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1