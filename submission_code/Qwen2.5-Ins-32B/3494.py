from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts to ensure they are in order
        horizontalCut.sort()
        verticalCut.sort()
        
        # Calculate the cost for horizontal cuts
        horizontal_cost = sum(horizontalCut) * n
        
        # Calculate the cost for vertical cuts
        vertical_cost = sum(verticalCut) * m
        
        return horizontal_cost + vertical_cost