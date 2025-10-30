class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all powers of 5 that can be represented with up to 15 bits
        powers = set()
        power = 1
        while power < (1 << 15):
            powers.add(power)
            power *= 5
        
        n = len(s)
        # dp[i] will store the minimum number of beautiful substrings for s[0..i-1]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue
            if s[i] == '0':
                continue
            current_num = 0
            for j in range(i, n):
                current_num = (current_num << 1) | int(s[j])
                if current_num in powers:
                    dp[j+1] = min(dp[j+1], dp[i] + 1)
        
        return dp[n] if dp[n] != float('inf') else -1