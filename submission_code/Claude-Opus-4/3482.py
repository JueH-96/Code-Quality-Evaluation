class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        n = len(target)
        # dp[i] = minimum cost to build target[0:i]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0  # Empty string costs 0
        
        # For each position in target
        for i in range(1, n + 1):
            # Try each word
            for j in range(len(words)):
                word = words[j]
                word_len = len(word)
                
                # Check if this word can end at position i
                if word_len <= i and target[i - word_len:i] == word:
                    # If we can use this word, update dp[i]
                    dp[i] = min(dp[i], dp[i - word_len] + costs[j])
        
        return dp[n] if dp[n] != float('inf') else -1