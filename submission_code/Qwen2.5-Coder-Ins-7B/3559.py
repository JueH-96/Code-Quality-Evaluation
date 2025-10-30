class Solution:
    def minValidStrings(self, words: List[str], target: str) -> int:
        from collections import defaultdict
        dp = defaultdict(lambda: float('inf'))
        dp[0] = 0
        
        for i in range(1, len(target) + 1):
            for word in words:
                if i >= len(word) and target[i - len(word):i] == word:
                    dp[i] = min(dp[i], dp[i - len(word)] + 1)
        
        return dp[len(target)] if dp[len(target)] != float('inf') else -1