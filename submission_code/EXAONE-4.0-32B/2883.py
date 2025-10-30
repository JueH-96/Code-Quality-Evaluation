class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        n = len(s)
        if s[0] == '0':
            return -1
        
        max_val = (1 << n) - 1
        powers_set = set()
        num = 1
        while num <= max_val:
            bin_str = bin(num)[2:]
            powers_set.add(bin_str)
            num *= 5
        
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            for j in range(i + 1, n + 1):
                substr = s[i:j]
                if substr[0] == '0':
                    break
                if substr in powers_set:
                    if dp[j] > dp[i] + 1:
                        dp[j] = dp[i] + 1
        
        return dp[n] if dp[n] != float('inf') else -1