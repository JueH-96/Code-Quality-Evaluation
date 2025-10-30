class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary representations of powers of 5 up to 15 bits
        valid = set()
        power = 1
        while True:
            binary = bin(power)[2:]  # Convert to binary without '0b' prefix
            if len(binary) > 15:
                break
            valid.add(binary)
            power *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: 0 characters need 0 partitions
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # Skip if current position is unreachable
            if s[i] == '0':
                continue  # Skip if starting with '0' as it can't be a valid substring
            
            # Check all possible substrings starting at index i
            for k in range(i, n):
                current_sub = s[i:k+1]
                if current_sub in valid:
                    if dp[k+1] > dp[i] + 1:
                        dp[k+1] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1