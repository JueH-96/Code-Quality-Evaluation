class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j, word in enumerate(words):
                if target[i-len(word):i] == word:
                    dp[i] = min(dp[i], dp[i-len(word)] + costs[j])
        
        return dp[n] if dp[n] != float('inf') else -1