class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            dp[i] = float('inf')
            for j in range(len(words)):
                word = words[j]
                cost = costs[j]
                if i >= len(word) and target[i-len(word):i] == word:
                    dp[i] = min(dp[i], dp[i-len(word)] + cost)
        
        if dp[n] == float('inf'):
            return -1
        else:
            return dp[n]