from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        horizontalCut.sort()
        verticalCut.sort()
        
        total_cost = 0
        for i in range(m-1):
            total_cost += horizontalCut[i]
        
        for j in range(n-1):
            total_cost += verticalCut[j]
        
        return total_cost