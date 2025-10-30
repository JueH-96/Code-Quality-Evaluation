class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        # dp[i] represents the minimum cost to form target[0:i]
        dp = [float('inf')] * (n + 1)
        # Base case: empty string costs 0 to form
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j, word in enumerate(words):
                word_len = len(word)
                # Check if the current word matches the suffix ending at position i
                if i >= word_len and target[i-word_len:i] == word:
                    # Update minimum cost
                    dp[i] = min(dp[i], dp[i - word_len] + costs[j])
        
        # If dp[n] is still infinity, it's impossible to form the target
        return dp[n] if dp[n] < float('inf') else -1