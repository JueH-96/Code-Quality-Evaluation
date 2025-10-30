class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string costs 0
        
        for i in range(n):
            if dp[i] == float('inf'):
                continue  # Can't reach this position
            
            for word, cost in zip(words, costs):
                word_len = len(word)
                # Check if word can be placed starting at position i
                if i + word_len <= n and target[i:i + word_len] == word:
                    dp[i + word_len] = min(dp[i + word_len], dp[i] + cost)
        
        return dp[n] if dp[n] != float('inf') else -1