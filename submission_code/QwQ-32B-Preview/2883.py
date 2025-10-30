class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute binary representations of powers of 5 up to length 15
        powers_of_five_binaries = {
            "1",       # 5^0
            "101",     # 5^1
            "11001",   # 5^2
            "1111101", # 5^3
            "1001110001", # 5^4
            "110001001001", # 5^5
            "11110100001001" # 5^6
        }
        
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string requires 0 partitions
        
        for i in range(n):
            for j in range(i, -1, -1):
                substring = s[j:i+1]
                if substring in powers_of_five_binaries:
                    dp[i+1] = min(dp[i+1], dp[j] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1