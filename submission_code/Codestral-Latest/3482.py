from typing import List
from collections import defaultdict

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Create a dictionary to store the minimum cost to form each prefix of the target string
        dp = defaultdict(lambda: float('inf'))
        dp[0] = 0  # Base case: cost to form an empty string is 0

        # Iterate through each character in the target string
        for i in range(1, len(target) + 1):
            # Iterate through each word and its corresponding cost
            for word, cost in zip(words, costs):
                # Check if the word can be a suffix of the current prefix of the target
                if target[:i].endswith(word):
                    # Update the minimum cost to form the current prefix
                    dp[i] = min(dp[i], dp[i - len(word)] + cost)

        # If dp[len(target)] is still infinity, it means it's impossible to form the target string
        return dp[len(target)] if dp[len(target)] != float('inf') else -1