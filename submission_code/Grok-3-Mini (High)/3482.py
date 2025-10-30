from typing import List

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Group words and find min cost for each unique word
        min_cost = {}
        for w, c in zip(words, costs):
            if w not in min_cost or c < min_cost[w]:
                min_cost[w] = c
        
        # dp[i] min cost to make target[0..i-1]
        dp = [float('inf')] * (len(target) + 1)
        dp[0] = 0
        
        for i in range(0, len(target) + 1):
            if dp[i] < float('inf'):
                for word, cost in min_cost.items():
                    len_w = len(word)
                    if i + len_w <= len(target) and all(target[i + c] == word[c] for c in range(len_w)):
                        dp[i + len_w] = min(dp[i + len_w], dp[i] + cost)
        
        if dp[-1] < float('inf'):
            return int(dp[-1])
        else:
            return -1