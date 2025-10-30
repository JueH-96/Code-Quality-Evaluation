class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for word in words:
                for j in range(1, len(word) + 1):
                    if j <= i and word[:j] == target[i-j:i]:
                        dp[i] = min(dp[i], dp[i-j] + 1)
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]