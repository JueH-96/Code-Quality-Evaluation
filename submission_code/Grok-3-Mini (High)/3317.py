from typing import List
from collections import Counter

class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        # Count the total frequency of each character
        char_count = Counter()
        for word in words:
            for c in word:
                char_count[c] += 1
        
        # Calculate P_total: sum of floor(count[c] / 2) for each c
        P_total = 0
        for count in char_count.values():
            P_total += count // 2
        
        # Calculate P_cost for each word: floor(len(word) / 2)
        P_cost_list = [len(word) // 2 for word in words]
        
        # Sort P_cost_list in ascending order
        P_cost_list.sort()
        
        # Greedy selection: add strings with smallest P_cost first
        sum_cost = 0
        num_pal = 0
        for cost in P_cost_list:
            if sum_cost + cost <= P_total:
                sum_cost += cost
                num_pal += 1
            else:
                break  # No need to check further since list is sorted
        
        return num_pal