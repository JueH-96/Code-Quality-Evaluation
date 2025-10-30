from typing import List

class Solution:
    def maximumHappinessSum(self, happiness: List[int], k: int) -> int:
        happiness.sort(reverse=True)
        total_happiness = 0
        decrement = 0
        
        for i in range(k):
            total_happiness += max(happiness[i] - decrement, 0)
            decrement += 1
        
        return total_happiness