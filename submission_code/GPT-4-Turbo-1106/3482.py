from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Create a dictionary to store the cost of each word
        word_cost = {word: cost for word, cost in zip(words, costs)}
        
        # Initialize a dictionary to store the minimum cost to reach each substring of target
        dp = {"": 0}
        
        for i in range(len(target)):
            for word in words:
                # If the word matches the end of the current substring of target
                if target.startswith(word, i - len(word) + 1):
                    # Get the substring of target before the word
                    prev = target[:i - len(word) + 1]
                    # If the previous substring is in dp, update the minimum cost for the current substring
                    if prev in dp:
                        curr = target[:i + 1]
                        if curr not in dp or dp[prev] + word_cost[word] < dp[curr]:
                            dp[curr] = dp[prev] + word_cost[word]
        
        # Return the minimum cost to reach the entire target string, or -1 if it's not possible
        return dp.get(target, -1)