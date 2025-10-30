from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts in descending order
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # Initialize the total cost
        total_cost = 0
        
        # Initialize the current height and width
        height = m
        width = n
        
        # Perform the cuts
        for i in range(m - 1):
            total_cost += horizontalCut[i] * width
            height -= 1
        for j in range(n - 1):
            total_cost += verticalCut[j] * height
            width -= 1
        
        return total_cost