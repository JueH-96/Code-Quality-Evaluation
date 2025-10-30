class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        valid = set()
        current = 1
        while True:
            binary_str = bin(current)[2:]
            if len(binary_str) > 15:
                break
            valid.add(binary_str)
            current *= 5
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                substr = s[j:i]
                if substr in valid:
                    dp[i] = min(dp[i], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1