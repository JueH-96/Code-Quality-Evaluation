class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all binary representations of 5^k up to 15 bits
        beautiful_substrings = set()
        current = 1
        while True:
            binary = bin(current)[2:]  # Convert to binary without '0b' prefix
            if len(binary) > 15:
                break
            beautiful_substrings.add(binary)
            current *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 partitions
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            # Check all possible substrings starting at i, up to the maximum possible length
            max_length = min(15, n - i)
            for l in range(1, max_length + 1):
                end = i + l
                substr = s[i:end]
                if substr in beautiful_substrings:
                    if dp[end] > dp[i] + 1:
                        dp[end] = dp[i] + 1
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]