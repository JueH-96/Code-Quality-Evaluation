from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # Sort the cuts to process them in order
        horizontalCut.sort()
        verticalCut.sort()
        
        # Initialize the total cost
        total_cost = 0
        
        # Add the edges of the cake to the cuts
        horizontalCut = [0] + horizontalCut + [m]
        verticalCut = [0] + verticalCut + [n]
        
        # Calculate the maximum height and width of the pieces
        max_height = max(horizontalCut[i] - horizontalCut[i-1] for i in range(1, len(horizontalCut)))
        max_width = max(verticalCut[i] - verticalCut[i-1] for i in range(1, len(verticalCut)))
        
        # The total cost is the sum of the maximum height times the vertical cuts and the maximum width times the horizontal cuts
        total_cost = max_height * sum(verticalCut[1:-1]) + max_width * sum(horizontalCut[1:-1])
        
        return total_cost