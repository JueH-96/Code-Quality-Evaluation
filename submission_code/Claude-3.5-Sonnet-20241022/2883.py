class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        if s[0] == '0':
            return -1
            
        powers = {bin(5**i)[2:] for i in range(7)}
        n = len(s)
        
        # dp[i] represents min partitions needed for s[i:]
        dp = [float('inf')] * (n + 1)
        dp[n] = 0
        
        # For each starting position
        for i in range(n-1, -1, -1):
            if s[i] == '0':
                continue
                
            # Try all possible substrings starting at i
            curr = ''
            for j in range(i, n):
                curr += s[j]
                if curr in powers:
                    dp[i] = min(dp[i], 1 + dp[j+1])
                    
        return dp[0] if dp[0] != float('inf') else -1