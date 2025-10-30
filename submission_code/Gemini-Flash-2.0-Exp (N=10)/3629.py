class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        counts = [0] * 26
        for char in s:
            counts[ord(char) - ord('a')] += 1
        
        dp = [0] * 26
        for i in range(26):
            dp[i] = counts[i]
        
        for _ in range(t):
            new_dp = [0] * 26
            for i in range(25):
                new_dp[i+1] += dp[i]
            new_dp[0] += dp[25]
            new_dp[1] += dp[25]
            dp = new_dp
            
        total_length = 0
        for i in range(26):
            total_length = (total_length + dp[i]) % MOD
        
        return total_length