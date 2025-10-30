from typing import List

class Solution:
    def minimumCost(self, m: int, n: int, horizontalCut: List[int], verticalCut: List[int]) -> int:
        # sort the cost arrays in descending order for maximum cost first 
        horizontalCut.sort(reverse=True)
        verticalCut.sort(reverse=True)
        
        # pieces count: at the beginning one piece of cake
        horizontalPieces = 1
        verticalPieces = 1
        
        i, j = 0, 0
        totalCost = 0
        
        # While there are still cuts left
        while i < len(horizontalCut) and j < len(verticalCut):
            # choose the larger cost among the next available horizontal and vertical cut
            if horizontalCut[i] >= verticalCut[j]:
                totalCost += horizontalCut[i] * verticalPieces  # cost is multiplied by current vertical segments
                horizontalPieces += 1  # increase the horizontal partition count
                i += 1
            else:
                totalCost += verticalCut[j] * horizontalPieces  # cost is multiplied by current horizontal segments
                verticalPieces += 1  # increase the vertical partition count
                j += 1
        
        # Process remaining horizontal cuts (if any)
        while i < len(horizontalCut):
            totalCost += horizontalCut[i] * verticalPieces
            i += 1
        
        # Process remaining vertical cuts (if any)
        while j < len(verticalCut):
            totalCost += verticalCut[j] * horizontalPieces
            j += 1
        
        return totalCost