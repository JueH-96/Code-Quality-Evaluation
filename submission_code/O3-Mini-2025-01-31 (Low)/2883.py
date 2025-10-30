class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute valid beautiful numbers as binary strings.
        # A beautiful number is a power of 5 with no leading zeros.
        # Since s length is at most 15, the maximum integer is 2**15 - 1.
        max_val = (1 << 15) - 1  # 32767
        valid_set = set()
        power = 1
        while power <= max_val:
            # Convert the power of 5 to its binary representation without the '0b' prefix.
            bin_str = bin(power)[2:]
            valid_set.add(bin_str)
            power *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            # Try every possible substring starting from i
            for j in range(i+1, n+1):
                substring = s[i:j]
                # Skip substrings with leading zeros
                if substring[0] == '0':
                    continue
                if substring in valid_set:
                    dp[j] = min(dp[j], dp[i] + 1)
                    
        return dp[n] if dp[n] != float('inf') else -1