from typing import List
from collections import defaultdict

class Solution:
    def minimumCost(self, target: str, words: List[str], costs: List[int]) -> int:
        # Create a dictionary to store the minimum cost for each prefix of the target
        min_cost = defaultdict(lambda: float('inf'))
        min_cost[""] = 0  # Base case: cost to form an empty string is 0
        
        # Iterate through each word and its corresponding cost
        for word, cost in zip(words, costs):
            # Update the minimum cost for each prefix of the target
            for prefix in list(min_cost.keys()):
                new_string = prefix + word
                if len(new_string) <= len(target) and target.startswith(new_string):
                    min_cost[new_string] = min(min_cost[new_string], min_cost[prefix] + cost)
        
        # The answer is the minimum cost to form the entire target string
        return min_cost[target] if min_cost[target] != float('inf') else -1