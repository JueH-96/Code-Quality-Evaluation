class Solution:
    def minimumBeautifulSubstrings(self, s: str) -> int:
        # Precompute all beautiful substrings
        beautiful = [
            '1',
            '101',
            '11001',
            '1111101',
            '1001110001',
            '1100001110001'
        ]
        n = len(s)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Base case: empty string needs 0 partitions
        
        for i in range(n):
            for t in beautiful:
                len_t = len(t)
                if i + 1 >= len_t:
                    j = i - len_t + 1
                    if j >= 0:
                        if s[j:i+1] == t:
                            if dp[j] + 1 < dp[i+1]:
                                dp[i+1] = dp[j] + 1
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]